import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import clip
from src.losses import build_text_anchor, PrototypeBank, classification_loss, structural_losses, prototype_contrastive_loss, nt_xent
from src.eval import compute_retrieval_metrics, get_metric_config

# def freeze_all_but_ln(module):
#     """Freeze an encoder, then enable only its LayerNorm parameters."""
#     module.requires_grad_(False)
#     for child in module.modules():
#         if isinstance(child, torch.nn.LayerNorm):
#             child.requires_grad_(True)

def freeze_all_but_ln(module):
    module.requires_grad_(True)

    for child in module.modules():
        if not isinstance(child, torch.nn.LayerNorm):
            if hasattr(child, "weight") and child.weight is not None:
                child.weight.requires_grad_(False)

            if hasattr(child, "bias") and child.bias is not None:
                child.bias.requires_grad_(False)

def print_trainable_parameters(model):
    """Print every parameter that will be updated by the optimizer."""
    trainable = []
    frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, tuple(param.shape), param.numel()))
        else:
            frozen += param.numel()

    print('\nTrainable parameters:')
    for name, shape, count in trainable:
        print(f'  {name:60s} shape={str(shape):18s} numel={count:,}')
    trainable_count = sum(count for _, _, count in trainable)
    total_count = trainable_count + frozen
    print(f'Trainable: {trainable_count:,} / {total_count:,} '
          f'({100.0 * trainable_count / total_count:.2f}%)\n')

class SGSPLModel(pl.LightningModule):

    def __init__(self, opts, seen_class_names: list):

        super().__init__()
        self.opts = opts
        self.seen_class_names = seen_class_names
        self.n_seen = len(seen_class_names)

        clip_model, _ = clip.load(opts.clip_model, device='cpu')
        clip_model.requires_grad_(False)

        self.clip = clip_model
        self.sketch_visual = copy.deepcopy(clip_model.visual)
        freeze_all_but_ln(self.clip.visual)
        freeze_all_but_ln(self.sketch_visual)

        # Frozen CLIP supplies the text anchor.
        self.clip_frozen = copy.deepcopy(clip_model)
        self.clip_frozen.requires_grad_(False)
        self.clip_frozen.eval()

        # Learnable prompt tokens (CLIP-AT style)
        visual_width = clip_model.visual.positional_embedding.shape[-1]
        self.sk_prompt  = nn.Parameter(torch.randn(opts.n_prompts, visual_width))
        self.img_prompt = nn.Parameter(torch.randn(opts.n_prompts, visual_width))

        # print_trainable_parameters(self)
        # Triplet loss (CLIP-AT baseline)
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_tri = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn,
            margin=opts.triplet_margin,
        )

        # EMA prototype bank
        self.bank = PrototypeBank(
            n_classes = self.n_seen,
            embed_dim = opts.embed_dim,
            momentum  = opts.ema_m,
        )

        # Text anchor A  (built after device is known → deferred)
        # Will be populated in setup() or first training_step via _ensure_anchor()
        self.register_buffer('anchor_A',       None, persistent=False)
        self.register_buffer('text_emb_seen',  None, persistent=False)
        self._anchor_built = False

        self.best_map  = -1.0

        # Collect ZS/GZS validation outputs across batches (Lightning 2.x)
        self._val_sk_feats  = []
        self._val_ph_feats  = []
        self._val_sk_labels = []
        self._val_ph_labels = []

    # Anchor matrix (deferred — needs device)
    def _ensure_anchor(self):
        """Build text anchor matrix on the correct device (called lazily)."""
        if self._anchor_built:
            return
        text_emb, anchor_A = build_text_anchor(
            clip_model   = self.clip_frozen,
            class_names  = self.seen_class_names,
            templates    = self.opts.text_templates,
            device       = self.device,
        )
        self.text_emb_seen = text_emb.to(self.device)
        self.anchor_A      = anchor_A.to(self.device)
        # Share anchor with bank (same object)
        self.bank.to(self.device)
        self._anchor_built = True


    def on_train_epoch_start(self):
        # re-lock clip_frozen to eval
        self.clip_frozen.eval()

    def forward(self, images: torch.Tensor, modality: str) -> torch.Tensor:
        prompt = self.sk_prompt if modality == 'sketch' else self.img_prompt
        if prompt.shape[0] == 0:
            prompt = None

        if modality == 'sketch':
            feats = self.sketch_visual(images.type(self.clip.dtype), prompt=prompt.expand(images.shape[0], -1, -1).type(self.clip.dtype))
        else:
            feats = self.clip.encode_image(images.type(self.clip.dtype), prompt=prompt.expand(images.shape[0], -1, -1).type(self.clip.dtype))

        feats = feats.float()                          # fp32 for stable loss
        return F.normalize(feats, dim=-1)

    def training_step(self, batch, batch_idx):
        self._ensure_anchor()

        sk, img, neg, cat_idx = batch
        # cat_idx: [B] integer indices into self.seen_class_names

        # Encode with prompted CLIP
        sk_feat  = self.forward(sk,  modality='sketch')    # [B, D]
        ph_feat  = self.forward(img, modality='image')     # [B, D]
        neg_feat = self.forward(neg, modality='image')     # [B, D]

        # Triplet loss (CLIP-AT baseline)
        loss_tri = self.loss_tri(sk_feat, ph_feat, neg_feat)

        # L_cls — classification loss
        logit_scale = self.clip.logit_scale.exp()
        loss_cls = classification_loss(
            sk_feat       = sk_feat,
            ph_feat       = ph_feat,
            cat_idx       = cat_idx,
            text_emb_seen = self.text_emb_seen,
            logit_scale   = logit_scale,
        )

        # Read the previous EMA state before this batch updates the bank.
        loss_proto = prototype_contrastive_loss(
            sk_feat=sk_feat,
            ph_feat=ph_feat,
            cat_idx=cat_idx,
            bank=self.bank,
            temperature=self.opts.proto_temp,
            reverse_weight=self.opts.proto_reverse_weight,
            warmup=self.opts.bank_warmup,
        )

        # Structural losses keep their existing update order and gradient path.
        self.bank.update(sk_feat.detach(),  cat_idx, modality='sk')
        self.bank.update(ph_feat.detach(), cat_idx, modality='ph')

        # L_SSC + L_xmod
        loss_ssc, loss_xmod = structural_losses(
            sk_feat  = sk_feat,
            ph_feat  = ph_feat,
            cat_idx  = cat_idx,
            bank     = self.bank,
            anchor_A = self.anchor_A,
            dist     = self.opts.ssc_dist,
            T        = self.opts.ssc_temp,
            warmup   = self.opts.bank_warmup,
            no_proto_grad = self.opts.no_proto_grad,
        )

        nt_xent_loss = nt_xent(sk_feat, ph_feat)

        # Total loss
        loss = (
            self.opts.triplet_weight * loss_tri
            + self.opts.classification_weight * loss_cls
            + self.opts.ssc_weight * (loss_ssc + self.opts.xmod_weight * loss_xmod)
            + self.opts.proto_weight * loss_proto
            + self.opts.nt_xent_weight * nt_xent_loss
        )

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_loss_proto', loss_proto, on_step=False, on_epoch=True)
        self.log('train_active_prototypes', self.bank.proto_mask.sum(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, cat_idx = batch

        if dataloader_idx == 0:
            feats = self.forward(imgs, modality='sketch')
            self._val_sk_feats.append(feats.cpu())
            self._val_sk_labels.append(cat_idx.cpu())
        else:
            feats = self.forward(imgs, modality='image')
            self._val_ph_feats.append(feats.cpu())
            self._val_ph_labels.append(cat_idx.cpu())

    def on_validation_epoch_end(self):
        if not self._val_sk_feats:
            return

        sk_feats  = torch.cat(self._val_sk_feats)
        ph_feats  = torch.cat(self._val_ph_feats)
        sk_labels = torch.cat(self._val_sk_labels)
        ph_labels = torch.cat(self._val_ph_labels)

        metric_cfg = get_metric_config(self.opts.dataset)

        # ZS/GZS-SBIR
        metrics = compute_retrieval_metrics(
            sk_feats  = sk_feats,
            ph_feats  = ph_feats,
            sk_labels = sk_labels,
            ph_labels = ph_labels,
            **metric_cfg,
        )
        map_k = metric_cfg['map_k']
        prec_k = metric_cfg['prec_k']
        map_score = metrics['mAP']
        prec_score = metrics['precision']

        if not self.trainer.sanity_checking and map_score > self.best_map:
            self.best_map = map_score
        
        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()

        self.log('mAP', map_score, prog_bar=False, on_epoch=True)     
        self.log(f'precision', prec_score, prog_bar=False, on_epoch=True)
        print(f"\nmAP@{map_k if map_k is not None else 'all'}: {map_score:.3f}, P@{prec_k}: {prec_score:.3f}, Best mAP: {self.best_map:.3f}")
        print(f"Train loss (epoch avg): {train_loss:.6f}")

        self._val_sk_feats.clear()
        self._val_ph_feats.clear()
        self._val_sk_labels.clear()
        self._val_ph_labels.clear()

    # Optimiser — two param groups with different LRs
    def configure_optimizers(self):
        """
        Prompt parameters:   lr_prompt (high LR — these are learnable from scratch)
        LayerNorm parameters: lr_ln    (low  LR — fine-tune pretrained LN stats)
        """
        prompt_params = [self.sk_prompt, self.img_prompt]

        ln_params = [
            parameter
            for encoder in (self.clip.visual, self.sketch_visual)
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ]

        # self.clip.logit_scale.requires_grad_(True)
        # if not any(p is self.clip.logit_scale for p in ln_params):
        #     ln_params.append(self.clip.logit_scale)

        # optimizer = torch.optim.Adam([
        #     {'params': prompt_params, 'lr': self.opts.lr_prompt},
        #     {'params': ln_params,     'lr': self.opts.lr_ln},
        # ], weight_decay=self.opts.weight_decay)

        optimizer = torch.optim.Adam([
            {'params': prompt_params, 'lr': self.opts.lr_prompt},
            {'params': ln_params,     'lr': self.opts.lr_ln},
        ])

        # Cosine LR schedule with linear warmup
        # total_steps   = self.trainer.estimated_stepping_batches
        # warmup_steps  = int(total_steps * self.opts.warmup_epochs / self.opts.max_epochs)

        # def lr_lambda(step):
        #     if step < warmup_steps:
        #         return float(step) / max(1, warmup_steps)
        #     progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        #     return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # return {
        #     'optimizer':  optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'interval':  'step',
        #     },
        # }
        return optimizer

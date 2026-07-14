"""
SG-SPL Lightning Module
=======================
Structure- & Geometry-regularized Prompt Learning for ZS-SBIR.

Architecture:
  - CLIP ViT-B/32 backbone (freeze all except LayerNorm, following CLIP-AT)
  - Two learnable prompt vectors: sk_prompt [n_prompts, D], img_prompt [n_prompts, D]
  - Frozen CLIP copy for text anchor + L_asym_sph reference
  - EMA prototype bank for L_SSC and L_xmod

Total loss:
  L = L_triplet
    + λ_cls  · L_cls
    + λ_ssc  · (L_SSC + λ_x · L_xmod)
    + L_asym_sph  (λ_ph and λ_sk are inside asym_spherical_loss)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import clip as clip_module
from src.losses import build_text_anchor, PrototypeBank, classification_loss, structural_losses, asym_spherical_loss
from src.eval import compute_retrieval_metrics, get_metric_config


def freeze_all_but_ln(module: nn.Module):
    for m in module.modules():
        if not isinstance(m, nn.LayerNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)


class SGSPLModel(pl.LightningModule):

    def __init__(self, opts, seen_class_names: list):
        """
        Args:
            opts:              argparse namespace (from experiments/options.py)
            seen_class_names:  list of seen (train) class name strings
        """
        super().__init__()
        self.opts = opts
        self.seen_class_names = seen_class_names
        self.n_seen = len(seen_class_names)

        # clip for prompt tuning (trainable except LayerNorm)
        clip_model, _ = clip_module.load(opts.clip_model, device='cpu')
        freeze_all_but_ln(clip_model)
        
        if opts.independent_ln:
            self.clip_sk = clip_model
            self.clip_ph = copy.deepcopy(clip_model)

            # Weight tying: share everything EXCEPT LayerNorms
            def tie_non_ln_weights(mod_sk, mod_ph):
                if not isinstance(mod_sk, nn.LayerNorm):
                    for name, param_sk in mod_sk.named_parameters(recurse=False):
                        setattr(mod_ph, name, param_sk)
                for name, child_sk in mod_sk.named_children():
                    tie_non_ln_weights(child_sk, getattr(mod_ph, name))
                    
            tie_non_ln_weights(self.clip_sk, self.clip_ph)
        else:
            self.clip_sk = clip_model
            self.clip_ph = clip_model

        # frozen clip for anchor + L_asym_sph
        self.clip_frozen = copy.deepcopy(clip_model)
        self.clip_frozen.requires_grad_(False)
        self.clip_frozen.eval()

        # Learnable prompt tokens (CLIP-AT style)
        visual_width = clip_model.visual.positional_embedding.shape[-1]
        self.sk_prompt  = nn.Parameter(torch.randn(opts.n_prompts, visual_width))
        self.img_prompt = nn.Parameter(torch.randn(opts.n_prompts, visual_width))

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

        self.best_zs_map  = -1.0
        self.best_gzs_map = -1.0

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
        """
        Encode images using the modality-specific prompt.

        Args:
            images:   [B, 3, H, W]
            modality: 'sketch' or 'image'

        Returns:
            features: [B, D] — L2-normalised embeddings
        """
        prompt = self.sk_prompt if modality == 'sketch' else self.img_prompt
        if prompt.shape[0] == 0:
            prompt = None
            
        clip_branch = self.clip_sk if modality == 'sketch' else self.clip_ph
        feats  = clip_branch.encode_image(images, prompt=prompt.expand(images.shape[0], -1, -1))
        feats  = feats.float()                          # fp32 for stable loss
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def _encode_frozen(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode with frozen (no-prompt) CLIP → used as anchor for L_asym_sph.
        Returns L2-normalised fp32 features.
        """
        feats = self.clip_frozen.encode_image(images, prompt=None)
        return F.normalize(feats.float(), dim=-1)


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
        logit_scale = self.clip_sk.logit_scale.exp()
        loss_cls = classification_loss(
            sk_feat       = sk_feat,
            ph_feat       = ph_feat,
            cat_idx       = cat_idx,
            text_emb_seen = self.text_emb_seen,
            logit_scale   = logit_scale,
        )

        #  Update EMA prototype bank (no grad)
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

        # L_asym_sph — frozen anchors (precomputed / on-the-fly)
        with torch.no_grad():
            sk_anchor = self._encode_frozen(sk)
            ph_anchor = self._encode_frozen(img)

        loss_sph = asym_spherical_loss(
            sk_feat   = sk_feat,
            ph_feat   = ph_feat,
            sk_anchor = sk_anchor,
            ph_anchor = ph_anchor,
            l_sph_ph  = self.opts.l_sph_ph,
            l_sph_sk  = self.opts.l_sph_sk,
        )

        # Total loss
        loss = (
            loss_tri
            + self.opts.l_cls * loss_cls
            + self.opts.l_ssc * (loss_ssc + self.opts.l_x * loss_xmod)
            + loss_sph
        )

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    # ──────────────────────────────────────────────────────────────────────────
    # Validation step — collect features
    # ──────────────────────────────────────────────────────────────────────────

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
        """Compute mAP and P@K from collected validation features."""
        if not self._val_sk_feats:
            return

        sk_feats  = torch.cat(self._val_sk_feats)
        ph_feats  = torch.cat(self._val_ph_feats)
        sk_labels = torch.cat(self._val_sk_labels)
        ph_labels = torch.cat(self._val_ph_labels)

        metric_cfg = get_metric_config(self.opts.dataset)

        # ZS-SBIR
        zs_metrics = compute_retrieval_metrics(
            sk_feats  = sk_feats,
            ph_feats  = ph_feats,
            sk_labels = sk_labels,
            ph_labels = ph_labels,
            **metric_cfg,
        )
        map_k = metric_cfg['map_k']
        prec_k = metric_cfg['prec_k']
        zs_map = zs_metrics['mAP']
        zs_prec = zs_metrics['precision']

        if zs_map > self.best_zs_map:
            self.best_zs_map = zs_map
        
        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()

        self.log('mAP', zs_map, prog_bar=False, on_epoch=True)     
        self.log(f'precision', zs_prec, prog_bar=False, on_epoch=True)
        print(f"\nmAP@{map_k if map_k is not None else 'all'}: {zs_map:.3f}, P@{prec_k}: {zs_prec:.3f}, Best mAP: {self.best_zs_map:.4f}")
        print(f"Train loss (epoch avg): {train_loss:.6f}")

        # Clear buffers
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
        ln_params = []
        for name, p in self.clip_sk.named_parameters():
            if p.requires_grad: ln_params.append(p)
        if self.opts.independent_ln:
            for name, p in self.clip_ph.named_parameters():
                if p.requires_grad: ln_params.append(p)

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

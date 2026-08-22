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

import clip
from src.losses import build_text_anchor, PrototypeBank, classification_loss, structural_losses, asym_spherical_loss
from src.eval import compute_retrieval_metrics, get_metric_config


# def freeze_all_but_ln(module: nn.Module):
#     for m in module.modules():
#         if not isinstance(m, nn.LayerNorm):
#             for p in m.parameters(recurse=False):
#                 p.requires_grad_(False)

def freeze_all_but_ln(module):
    """Freeze an encoder, then enable only its LayerNorm parameters."""
    module.requires_grad_(False)
    for child in module.modules():
        if isinstance(child, torch.nn.LayerNorm):
            child.requires_grad_(True)

class SGSPLModel(pl.LightningModule):

    def __init__(self, opts, seen_class_names: list):

        super().__init__()
        self.opts = opts
        self.seen_class_names = seen_class_names
        self.n_seen = len(seen_class_names)

        clip_model, _ = clip.load(opts.clip_model, device='cpu')
        clip_model.requires_grad_(False)

        if opts.independent_ln:
            
            self.clip_sk = clip_model
            self.clip_ph = copy.deepcopy(clip_model)
            freeze_all_but_ln(self.clip_sk.visual)
            freeze_all_but_ln(self.clip_ph.visual)
            print("[CLIP] separate_visual=True  → sk and ph have independent visual encoders")
        else:
            
            self.clip_sk = clip_model
            self.clip_ph = clip_model
            freeze_all_but_ln(self.clip_sk.visual)
            print("[CLIP] separate_visual=False → sk and ph share one visual encoder")

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

        # Gom feature validation qua các batch (Lightning 2.x).
        # 4 nhóm theo dataloader_idx (xem train.py):
        #   unseen: query = sketch, gallery = photo  → ZS-SBIR
        #   seen  : chỉ dùng khi bật --eval gzs      → GZS + generalization gap
        self._u_sk_f, self._u_sk_l = [], []   # unseen sketch (query)
        self._u_ph_f, self._u_ph_l = [], []   # unseen photo  (gallery)
        self._s_sk_f, self._s_sk_l = [], []   # seen sketch   (query)
        self._s_ph_f, self._s_ph_l = [], []   # seen photo    (gallery)

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
            l_sph_ph  = self.opts.sph_ph_weight,
            l_sph_sk  = self.opts.sph_sk_weight,
        )

        # Total loss
        loss = (
            self.opts.triplet_weight * loss_tri
            + self.opts.classification_weight * loss_cls
            + self.opts.ssc_weight * (loss_ssc + self.opts.xmod_weight * loss_xmod)
            + loss_sph
        )

        # Log từng loss để nhìn thấy cân bằng scale.
        #  - loss/*   : giá trị GỐC của mỗi loss (chưa nhân trọng số)
        #  - contrib/*: phần ĐÓNG GÓP thực tế vào tổng (đã nhân trọng số)
        # So sánh contrib/* cho biết loss nào đang chi phối gradient.
        self.log_dict({
            'loss/triplet': loss_tri,
            'loss/cls':     loss_cls,
            'loss/ssc':     loss_ssc,
            'loss/xmod':    loss_xmod,
            'loss/sph':     loss_sph,
            'contrib/triplet': self.opts.triplet_weight * loss_tri,
            'contrib/cls':     self.opts.classification_weight * loss_cls,
            'contrib/ssc':     self.opts.ssc_weight * loss_ssc,
            'contrib/xmod':    self.opts.ssc_weight * self.opts.xmod_weight * loss_xmod,
            'contrib/sph':     loss_sph,
        }, on_step=False, on_epoch=True)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    # ──────────────────────────────────────────────────────────────────────────
    # Validation step — collect features
    # ──────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, cat_idx = batch

        # Thứ tự loader khớp với train.py:
        #   0 = sketch unseen, 1 = photo unseen, 2 = sketch seen, 3 = photo seen
        if dataloader_idx == 0:
            f = self.forward(imgs, modality='sketch')
            self._u_sk_f.append(f.cpu());  self._u_sk_l.append(cat_idx.cpu())
        elif dataloader_idx == 1:
            f = self.forward(imgs, modality='image')
            self._u_ph_f.append(f.cpu());  self._u_ph_l.append(cat_idx.cpu())
        elif dataloader_idx == 2:
            f = self.forward(imgs, modality='sketch')
            self._s_sk_f.append(f.cpu());  self._s_sk_l.append(cat_idx.cpu())
        else:
            f = self.forward(imgs, modality='image')
            self._s_ph_f.append(f.cpu());  self._s_ph_l.append(cat_idx.cpu())

    def on_validation_epoch_end(self):
        """Tính mAP/P@K từ feature đã gom. ZS luôn tính; GZS + gap chỉ khi có seen."""
        if not self._u_sk_f:
            return

        metric_cfg = get_metric_config(self.opts.dataset)
        map_k  = metric_cfg['map_k']
        prec_k = metric_cfg['prec_k']

        sk_u  = torch.cat(self._u_sk_f);  skl_u = torch.cat(self._u_sk_l)
        ph_u  = torch.cat(self._u_ph_f);  phl_u = torch.cat(self._u_ph_l)

        # ── ZS-SBIR: query = sketch unseen, gallery = photo unseen ──────────────
        zs = compute_retrieval_metrics(sk_u, ph_u, skl_u, phl_u, **metric_cfg)
        zs_map, zs_prec = zs['mAP'], zs['precision']
        if zs_map > self.best_zs_map:
            self.best_zs_map = zs_map

        self.log('mAP', zs_map, prog_bar=False, on_epoch=True)
        self.log('precision', zs_prec, prog_bar=False, on_epoch=True)
        print(f"\n[ZS ] mAP@{map_k if map_k is not None else 'all'}: {zs_map:.3f}, "
              f"P@{prec_k}: {zs_prec:.3f}, Best: {self.best_zs_map:.3f}")

        # ── GZS-SBIR + generalization gap (chỉ khi bật --eval gzs) ──────────────
        if self._s_ph_f:
            ph_all  = torch.cat([ph_u,  torch.cat(self._s_ph_f)])
            phl_all = torch.cat([phl_u, torch.cat(self._s_ph_l)])

            # GZS: query unseen, gallery = seen + unseen (seen là distractor).
            gzs_u = compute_retrieval_metrics(sk_u, ph_all, skl_u, phl_all, **metric_cfg)
            gzs_map = gzs_u['mAP']
            if gzs_map > self.best_gzs_map:
                self.best_gzs_map = gzs_map

            self.log('GZS_mAP', gzs_map, prog_bar=False, on_epoch=True)
            self.log('GZS_precision', gzs_u['precision'], prog_bar=False, on_epoch=True)

            line = (f"[GZS] mAP@{map_k if map_k is not None else 'all'}: {gzs_map:.3f}, "
                    f"P@{prec_k}: {gzs_u['precision']:.3f}, Best: {self.best_gzs_map:.3f}")

            # Generalization gap: cùng gallery, so query seen vs query unseen.
            if self._s_sk_f:
                sk_s  = torch.cat(self._s_sk_f);  skl_s = torch.cat(self._s_sk_l)
                gzs_s = compute_retrieval_metrics(sk_s, ph_all, skl_s, phl_all, **metric_cfg)
                gap = gzs_s['mAP'] - gzs_map      # seen mAP − unseen mAP
                self.log('GZS_seen_mAP', gzs_s['mAP'], prog_bar=False, on_epoch=True)
                self.log('gen_gap', gap, prog_bar=False, on_epoch=True)
                line += f" | seen mAP: {gzs_s['mAP']:.3f}, gap: {gap:.3f}"

            print(line)

        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()
        print(f"Train loss (epoch avg): {train_loss:.6f}")

        # Clear buffers
        for buf in (self._u_sk_f, self._u_sk_l, self._u_ph_f, self._u_ph_l,
                    self._s_sk_f, self._s_sk_l, self._s_ph_f, self._s_ph_l):
            buf.clear()

    # Optimiser — two param groups with different LRs
    def configure_optimizers(self):
        """
        Prompt parameters:   lr_prompt (high LR — these are learnable from scratch)
        LayerNorm parameters: lr_ln    (low  LR — fine-tune pretrained LN stats)
        """
        prompt_params = [self.sk_prompt, self.img_prompt]

        ln_params  = []
        seen_p_ids = set()
        for branch_visual in {id(self.clip_sk.visual): self.clip_sk.visual,
                              id(self.clip_ph.visual): self.clip_ph.visual}.values():
            for p in branch_visual.parameters():
                if p.requires_grad and id(p) not in seen_p_ids:
                    ln_params.append(p)
                    seen_p_ids.add(id(p))

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

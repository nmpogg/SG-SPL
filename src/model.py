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

import clip as clip_module                       # our clip/ package
from src.losses import (
    build_text_anchor,
    PrototypeBank,
    classification_loss,
    structural_losses,
    asym_spherical_loss,
)
from src.eval import compute_retrieval_metrics, get_metric_config


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def freeze_all_but_ln(module: nn.Module):
    """
    Freeze every parameter except those in LayerNorm layers.
    Matches the CLIP-AT strategy ('freeze_all_but_bn' in their code,
    which actually checks isinstance(m, LayerNorm)).
    """
    for m in module.modules():
        if not isinstance(m, nn.LayerNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)


# ─────────────────────────────────────────────────────────────────────────────
# SGSPLModel
# ─────────────────────────────────────────────────────────────────────────────

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

        # ── 1. CLIP backbone ──────────────────────────────────────────────
        clip_model, _ = clip_module.load(opts.clip_model, device='cpu')
        clip_model = clip_model.float()     # work in fp32 internally
        freeze_all_but_ln(clip_model)
        self.clip = clip_model

        # ── 2. Frozen CLIP copy (no grad, eval always) ────────────────────
        # deepcopy BEFORE any training modifies LayerNorm weights
        self.clip_frozen = copy.deepcopy(clip_model)
        self.clip_frozen.requires_grad_(False)
        self.clip_frozen.eval()

        # ── 3. Learnable prompt tokens (CLIP-AT style) ────────────────────
        # IMPORTANT: prompt dim = transformer INTERNAL width, NOT output embed_dim.
        #   ViT-B/32: width=768, output=512
        #   ViT-B/16: width=768, output=512
        #   ViT-L/14: width=1024, output=768
        # Infer from positional_embedding to be robust across all backbones.
        visual_width = clip_model.visual.positional_embedding.shape[-1]
        self.sk_prompt  = nn.Parameter(
            torch.randn(opts.n_prompts, visual_width) * 0.02
        )
        self.img_prompt = nn.Parameter(
            torch.randn(opts.n_prompts, visual_width) * 0.02
        )

        # ── 4. Triplet loss (CLIP-AT baseline) ────────────────────────────
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn,
            margin=opts.triplet_margin,
        )

        # ── 5. EMA prototype bank ─────────────────────────────────────────
        self.bank = PrototypeBank(
            n_classes = self.n_seen,
            embed_dim = opts.embed_dim,
            momentum  = opts.ema_m,
        )

        # ── 6. Text anchor A  (built after device is known → deferred) ────
        # Will be populated in setup() or first training_step via _ensure_anchor()
        self.register_buffer('anchor_A',       None, persistent=False)
        self.register_buffer('text_emb_seen',  None, persistent=False)
        self._anchor_built = False

        # ── Misc ──────────────────────────────────────────────────────────
        self.best_zs_map  = -1.0
        self.best_gzs_map = -1.0

        # Collect ZS/GZS validation outputs across batches (Lightning 2.x)
        self._val_sk_feats  = []
        self._val_ph_feats  = []
        self._val_sk_labels = []
        self._val_ph_labels = []

    # ──────────────────────────────────────────────────────────────────────────
    # Anchor matrix (deferred — needs device)
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # Lightning hooks
    # ──────────────────────────────────────────────────────────────────────────

    def on_train_epoch_start(self):
        # Lightning sets ALL sub-modules to train mode at epoch start.
        # We must re-lock clip_frozen to eval so BN/LN stats don't shift.
        self.clip_frozen.eval()

    # ──────────────────────────────────────────────────────────────────────────
    # Forward: encode a batch with the prompted CLIP
    # ──────────────────────────────────────────────────────────────────────────

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
        feats  = self.clip.encode_image(images, prompt=prompt)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        self._ensure_anchor()

        sk, img, neg, cat_name, cat_idx = batch
        # cat_idx: [B] integer indices into self.seen_class_names

        # ── Encode with prompted CLIP ──────────────────────────────────────
        sk_feat  = self.forward(sk,  modality='sketch')    # [B, D]
        ph_feat  = self.forward(img, modality='image')     # [B, D]
        neg_feat = self.forward(neg, modality='image')     # [B, D]

        # ── Triplet loss (CLIP-AT baseline) ───────────────────────────────
        loss_tri = self.loss_fn(sk_feat, ph_feat, neg_feat)

        # ── L_cls — classification loss ───────────────────────────────────
        loss_cls = classification_loss(
            sk_feat       = sk_feat,
            ph_feat       = ph_feat,
            cat_idx       = cat_idx,
            text_emb_seen = self.text_emb_seen,
            logit_scale   = 1.0 / 0.07,
        )

        # ── Update EMA prototype bank (no grad) ───────────────────────────
        self.bank.update(sk_feat.detach(),  cat_idx, modality='sk')
        self.bank.update(ph_feat.detach(), cat_idx, modality='ph')

        # ── L_SSC + L_xmod ────────────────────────────────────────────────
        loss_ssc, loss_xmod = structural_losses(
            sk_feat  = sk_feat,
            ph_feat  = ph_feat,
            cat_idx  = cat_idx,
            bank     = self.bank,
            anchor_A = self.anchor_A,
            dist     = self.opts.ssc_dist,
            T        = self.opts.ssc_temp,
            warmup   = self.opts.bank_warmup,
        )

        # ── L_asym_sph — frozen anchors (precomputed / on-the-fly) ────────
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

        # ── Total loss ────────────────────────────────────────────────────
        loss = (
            loss_tri
            + self.opts.l_cls * loss_cls
            + self.opts.l_ssc * (loss_ssc + self.opts.l_x * loss_xmod)
            + loss_sph
        )

        # ── Logging ───────────────────────────────────────────────────────
        self.log_dict({
            'train/loss_tri':   loss_tri,
            'train/loss_cls':   loss_cls,
            'train/loss_ssc':   loss_ssc,
            'train/loss_xmod':  loss_xmod,
            'train/loss_sph':   loss_sph,
            'train/loss_total': loss,
            'train/n_protos':   float(self.bank.proto_mask.sum().item()),
        }, on_step=True, on_epoch=True, prog_bar=False, batch_size=sk.size(0))

        return loss

    # ──────────────────────────────────────────────────────────────────────────
    # Validation step — collect features
    # ──────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Collect sketch and photo features for retrieval evaluation.
        We store two lists:
          dataloader_idx=0 → ZS-SBIR  (gallery = unseen photos only)
          dataloader_idx=1 → GZS-SBIR (gallery = seen + unseen photos)
        """
        sk, ph, _, _, cat_idx = batch

        sk_feat = self.forward(sk,  modality='sketch')
        ph_feat = self.forward(ph,  modality='image')

        if dataloader_idx == 0:
            self._val_sk_feats.append(sk_feat.cpu())
            self._val_sk_labels.append(cat_idx.cpu())
        # Photo gallery is the same for both ZS and GZS
        self._val_ph_feats.append(ph_feat.cpu())
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

        zs_map = zs_metrics['mAP']
        prec_k = metric_cfg['prec_k']

        # Only show mAP on progress bar to prevent text wrapping/breaking
        self.log('val/ZS_mAP', zs_map, prog_bar=True, on_epoch=True)
        self.log(f'val/ZS_P@{prec_k}', zs_metrics[f'P@{prec_k}'], prog_bar=False, on_epoch=True)

        if zs_map > self.best_zs_map:
            self.best_zs_map = zs_map
            self.log('val/best_ZS_mAP', self.best_zs_map, prog_bar=False)

        # Clear buffers
        self._val_sk_feats.clear()
        self._val_ph_feats.clear()
        self._val_sk_labels.clear()
        self._val_ph_labels.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Optimiser — two param groups with different LRs
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """
        Prompt parameters:   lr_prompt (high LR — these are learnable from scratch)
        LayerNorm parameters: lr_ln    (low  LR — fine-tune pretrained LN stats)
        """
        prompt_params = [self.sk_prompt, self.img_prompt]
        ln_params = [
            p for name, p in self.clip.named_parameters()
            if p.requires_grad  # only LayerNorm weights are unfrozen
        ]

        optimizer = torch.optim.AdamW([
            {'params': prompt_params, 'lr': self.opts.lr_prompt},
            {'params': ln_params,     'lr': self.opts.lr_ln,    'weight_decay': 0.0},
        ], weight_decay=self.opts.weight_decay)

        # Cosine LR schedule with linear warmup
        total_steps   = self.trainer.estimated_stepping_batches
        warmup_steps  = int(total_steps * self.opts.warmup_epochs / self.opts.max_epochs)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer':  optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval':  'step',
            },
        }

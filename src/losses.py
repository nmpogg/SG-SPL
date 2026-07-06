"""
SG-SPL Loss Functions
=====================
Ba thành phần regularizer giữ cấu trúc CLIP trong khi prompt tuning:

  L = L_triplet + λ_cls·L_cls
    + λ_ssc·(L_SSC + λ_x·L_xmod)
    + λ_sph_ph·L_ph + λ_sph_sk·L_sk

L_SSC     — Dual-modality Semantic Structure Consistency   (EBSeg → retrieval)
L_xmod    — Cross-modal Structure Consistency              (novel contribution)
L_asym_sph — Asymmetric Hyperspherical Anchoring           (PromptSRC → asymmetric)

Tham khảo:
  - EBSeg: Shan et al., CVPR 2024 (arxiv 2406.09829)
  - PromptSRC: Khattak et al., ICCV 2023 (arxiv 2307.06948)
  - Relational KD: Park et al., CVPR 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Anchor matrix builder
@torch.no_grad()
def build_text_anchor(clip_model, class_names, templates, device):
    """
    Build the text-based anchor matrix A ∈ R^{C_s × C_s} (offline, once).

    A[i,j] = cosine_similarity(text_emb_i, text_emb_j)

    Uses ensemble of multiple text templates for robustness.

    Args:
        clip_model:   frozen CLIP model (clip.model.CLIP)
        class_names:  list of C_s seen class name strings
        templates:    list of template strings with {} placeholder
        device:       torch device

    Returns:
        text_emb: [C_s, D] — normalised mean text embeddings
        anchor_A: [C_s, C_s] — cosine similarity matrix
    """
    from clip.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()

    emb_list = []
    for cls_name in class_names:
        # Encode each template and average
        prompts = [t.format(cls_name.replace('_', ' ')) for t in templates]
        # Tokenize
        tokens = torch.cat([
            _tokenize(p, tokenizer) for p in prompts
        ]).to(device)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens).float()   # [n_templates, D]
            feats = F.normalize(feats, dim=-1)
            feat_mean = F.normalize(feats.mean(0), dim=-1)   # [D]
        emb_list.append(feat_mean)

    text_emb = torch.stack(emb_list)                         # [C_s, D]
    anchor_A = text_emb @ text_emb.t()                       # [C_s, C_s]
    return text_emb, anchor_A


def _tokenize(text: str, tokenizer, context_length: int = 77):
    """Tokenise a single string → LongTensor [1, context_length]."""
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot] + tokenizer.encode(text) + [eot]
    result = torch.zeros(1, context_length, dtype=torch.long)
    n = min(len(tokens), context_length)
    result[0, :n] = torch.tensor(tokens[:n])
    result[0, n - 1] = eot
    return result # id token


# EMA Prototype Bank
class PrototypeBank(nn.Module):
    """
    EMA-updated per-class prototype bank for sketch and photo modalities.

    bank[c] ← normalize(m·bank[c] + (1−m)·batch_mean_c)

    Gradient trick: bank is updated with no_grad (statistics only).
    When computing loss, we substitute prototypes of in-batch classes
    with the differentiable batch mean → gradient flows back to prompts.
    """

    def __init__(self, n_classes: int, embed_dim: int, momentum: float = 0.9):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.momentum  = momentum

        self.register_buffer('proto_sk',   torch.zeros(n_classes, embed_dim))
        self.register_buffer('proto_ph',   torch.zeros(n_classes, embed_dim))
        self.register_buffer('proto_mask', torch.zeros(n_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, feats: torch.Tensor, cat_idx: torch.Tensor, modality: str):
        """
        Update bank for a single modality with current batch features.

        Args:
            feats:    [B, D] normalised features (detached)
            cat_idx:  [B]   integer class indices
            modality: 'sk' or 'ph'
        """
        bank = self.proto_sk if modality == 'sk' else self.proto_ph

        for c in cat_idx.unique():
            c = c.item()
            mask = (cat_idx == c)
            f = F.normalize(feats[mask].mean(0), dim=-1)   # [D]
            if self.proto_mask[c]:
                bank[c] = F.normalize(self.momentum * bank[c] + (1 - self.momentum) * f, dim=-1)
            else:
                bank[c] = f
            self.proto_mask[c] = True

    def get_prototypes(
        self,
        sk_feat: torch.Tensor,
        ph_feat: torch.Tensor,
        cat_idx: torch.Tensor,
        no_grad: bool = False,
    ):
        """
        Return prototype matrices Psk, Pph where:
          - in-batch classes use differentiable batch mean → gradient flows (to loss and prompts)
          - out-of-batch classes use detached EMA bank values

        Returns:
            Psk: [n_classes, D]
            Pph: [n_classes, D]
            idx: LongTensor of active prototype indices
        """
        Psk = self.proto_sk.clone()
        Pph = self.proto_ph.clone()

        if not no_grad:
            for c in cat_idx.unique():
                c_int = c.item()
                mask = (cat_idx == c_int)
                Psk[c_int] = F.normalize(sk_feat[mask].mean(0), dim=-1)
                Pph[c_int] = F.normalize(ph_feat[mask].mean(0), dim=-1)

        idx = self.proto_mask.nonzero(as_tuple=True)[0]
        return Psk, Pph, idx


# Classification Loss  L_cls
def classification_loss(
    sk_feat:       torch.Tensor,   # [B, D]
    ph_feat:       torch.Tensor,   # [B, D]
    cat_idx:       torch.Tensor,   # [B]
    text_emb_seen: torch.Tensor,   # [C_s, D]
    logit_scale:   float,
) -> torch.Tensor:
    """
    Symmetric image–text classification loss over seen classes.

    L_cls = 0.5 · (CE(sketch→text) + CE(photo→text))

    logit_scale is fixed (not learnable) to avoid instability.
    """
    sk_n  = F.normalize(sk_feat, dim=-1)
    ph_n  = F.normalize(ph_feat, dim=-1)
    text_n = F.normalize(text_emb_seen, dim=-1)   # already normalised, safety check

    logits_sk  = logit_scale * sk_n  @ text_n.t()   # [B, C_s]
    logits_ph  = logit_scale * ph_n  @ text_n.t()   # [B, C_s]

    loss = 0.5 * (
        F.cross_entropy(logits_sk,  cat_idx) +
        F.cross_entropy(logits_ph, cat_idx)
    )
    return loss


# Structural Losses  L_SSC + L_xmod
def structural_losses(
    sk_feat:  torch.Tensor,         # [B, D]
    ph_feat:  torch.Tensor,         # [B, D]
    cat_idx:  torch.Tensor,         # [B]
    bank:     PrototypeBank,
    anchor_A: torch.Tensor,         # [C_s, C_s]
    dist:     str   = 'mse',        # 'mse' | 'kl'
    T:        float = 0.1,
    warmup:   int   = 10,
    no_proto_grad: bool = False,
) -> tuple:
    """
    Compute L_SSC and L_xmod using the prototype bank.

    L_SSC  = D(S_sk, A) + D(S_ph, A)      -- each modality vs text anchor
    L_xmod = D(S_sk, stopgrad(S_ph))
           + D(S_ph, stopgrad(S_sk))       -- cross-modal alignment

    D = MSE (original EBSeg) or symmetric KL divergence (ablation option).

    Returns: (loss_ssc, loss_xmod)
    """
    Psk, Pph, idx = bank.get_prototypes(sk_feat, ph_feat, cat_idx, no_grad=no_proto_grad)

    # Warm-up guard: wait until enough classes have been seen
    if idx.numel() < warmup:
        zero = sk_feat.new_zeros(1).squeeze()
        return zero, zero

    A = anchor_A[idx][:, idx]          # [K, K]
    Ssk = Psk[idx] @ Psk[idx].t()      # [K, K]
    Sph = Pph[idx] @ Pph[idx].t()      # [K, K]

    if dist == 'mse':
        loss_ssc  = F.mse_loss(Ssk, A) + F.mse_loss(Sph, A)
        loss_xmod = (F.mse_loss(Ssk, Sph.detach()) +
                     F.mse_loss(Sph, Ssk.detach()))
    else:   # KL divergence
        def kl(P, Q):
            return F.kl_div(
                F.log_softmax(P / T, dim=-1),
                F.softmax(Q / T, dim=-1).detach(),
                reduction='batchmean'
            )
        loss_ssc  = kl(Ssk, A) + kl(Sph, A)
        loss_xmod = 0.5 * (kl(Ssk, Sph.detach()) + kl(Sph, Ssk.detach()))

    return loss_ssc, loss_xmod



# Asymmetric Hyperspherical Anchoring  L_asym_sph
def asym_spherical_loss(
    sk_feat:    torch.Tensor,   # [B, D] — prompted sketch features
    ph_feat:    torch.Tensor,   # [B, D] — prompted photo features
    sk_anchor:  torch.Tensor,   # [B, D] — frozen CLIP sketch features (no grad)
    ph_anchor:  torch.Tensor,   # [B, D] — frozen CLIP photo features  (no grad)
    l_sph_ph:   float = 1.0,    # λ_ph   — pull photo strongly (in-distribution)
    l_sph_sk:   float = 0.2,    # λ_sk   — pull sketch weakly  (out-of-distribution)
) -> torch.Tensor:
    """
    L_asym_sph = λ_ph · E_x[1 − cos(f_ph(x), f_frozen(x))]
               + λ_sk · E_x[1 − cos(f_sk(x), f_frozen(x))]

    Asymmetry rationale:
      - CLIP was pretrained on natural images → frozen photo anchor is reliable → pull hard
      - CLIP rarely saw hand-drawn sketches → frozen sketch anchor is unreliable → pull lightly
        (pulling too hard would prevent the sketch encoder from adapting)

    Difference from PromptSRC:
      PromptSRC: symmetric, per-sample, single-modal classification.
      SG-SPL:    asymmetric by modality, for cross-modal retrieval.
    """
    # Normalise all features to unit sphere (cosine similarity = dot product)
    sk_feat_n   = F.normalize(sk_feat.float(),   dim=-1)
    ph_feat_n   = F.normalize(ph_feat.float(),   dim=-1)
    sk_anchor_n = F.normalize(sk_anchor.float(), dim=-1)
    ph_anchor_n = F.normalize(ph_anchor.float(), dim=-1)

    l_ph = (1.0 - F.cosine_similarity(ph_feat_n, ph_anchor_n)).mean()
    l_sk = (1.0 - F.cosine_similarity(sk_feat_n, sk_anchor_n)).mean()

    return l_sph_ph * l_ph + l_sph_sk * l_sk

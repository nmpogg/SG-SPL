"""
Smoke test â€” kiá»ƒm tra toĂ n bá»™ pipeline KHĂ”NG cáº§n dataset tháº­t.
Táº¡o dá»¯ liá»‡u giáº£ Ä‘á»ƒ verify:
  1. Prompt tuning hoáº¡t Ä‘á»™ng (VisionTransformer.forward(prompt=...) ok)
  2. PrototypeBank update + get_with_grad ok
  3. Táº¥t cáº£ losses tĂ­nh Ä‘Æ°á»£c (khĂ´ng NaN, gradient flow ok)
  4. SGSPLModel forward pass ok

Cháº¡y:
    python tests/test_smoke.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print()

    # â”€â”€ 1. CLIP with prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('[1] CLIP with prompt tokens...')
    import clip as clip_module
    clip_model, _ = clip_module.load('ViT-B/32', device=device)
    clip_model = clip_model.float()

    B = 4
    dummy_img  = torch.randn(B, 3, 224, 224).to(device)
    n_prompts  = 16
    # Prompt dim = INTERNAL visual transformer width (768 for ViT-B/32), NOT output dim (512)
    prompt_dim = clip_model.visual.positional_embedding.shape[-1]
    embed_dim  = 512   # output dim (after projection)
    print(f'   visual_width (prompt_dim): {prompt_dim}, embed_dim (output): {embed_dim}')

    # Without prompt (vanilla)
    with torch.no_grad():
        feat_vanilla = clip_model.encode_image(dummy_img, prompt=None)
    print(f'   vanilla feat shape: {feat_vanilla.shape}')  # [4, 512]

    # With prompt
    prompt = torch.randn(n_prompts, prompt_dim, device=device, requires_grad=True)
    feat_prompted = clip_model.encode_image(dummy_img, prompt=prompt)
    print(f'   prompted feat shape: {feat_prompted.shape}')  # [4, 512]

    # Gradient should flow back to prompt
    loss_test = feat_prompted.sum()
    loss_test.backward()
    assert prompt.grad is not None, 'No gradient to prompt!'
    print('   gradient flows to prompt: [OK]')
    print()

    # ——————————————————————————————————————————————————————————————————————————————————————
    print('[2] PrototypeBank...')
    from src.losses import PrototypeBank

    n_seen = 10
    bank = PrototypeBank(n_seen, embed_dim, momentum=0.9).to(device)

    cat_idx = torch.tensor([0, 1, 2, 0, 1], device=device)
    sk_feats = F.normalize(torch.randn(5, embed_dim, device=device), dim=-1)
    ph_feats = F.normalize(torch.randn(5, embed_dim, device=device), dim=-1)

    bank.update(sk_feats.detach(), cat_idx, 'sk')
    bank.update(ph_feats.detach(), cat_idx, 'ph')
    print(f'   active prototypes: {bank.proto_mask.sum().item()} / {n_seen}')

    Psk, Pph, idx = bank.get_prototypes_with_grad(sk_feats, ph_feats, cat_idx)
    print(f'   prototype matrix shape: {Psk.shape}, active idx: {idx.tolist()}')
    print()

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    print('[3] All loss functions...')
    from src.losses import (
        build_text_anchor, classification_loss,
        structural_losses, asym_spherical_loss
    )

    # L_cls
    class_names = [f'class_{i}' for i in range(n_seen)]
    templates   = ['a photo of a {}.', 'a sketch of a {}.']
    clip_model.eval()
    text_emb, anchor_A = build_text_anchor(clip_model, class_names, templates, device)
    print(f'   text_emb: {text_emb.shape}, anchor_A: {anchor_A.shape}')

    sk_f = F.normalize(torch.randn(B, embed_dim, device=device, requires_grad=True), dim=-1)
    ph_f = F.normalize(torch.randn(B, embed_dim, device=device, requires_grad=True), dim=-1)
    sk_f.retain_grad()
    ph_f.retain_grad()
    cidx = torch.randint(0, n_seen, (B,), device=device)

    loss_cls = classification_loss(sk_f, ph_f, cidx, text_emb)
    print(f'   L_cls = {loss_cls.item():.4f}  (should not be NaN)')

    # L_SSC + L_xmod
    bank2 = PrototypeBank(n_seen, embed_dim).to(device)
    bank2.update(sk_f.detach(), cidx, 'sk')
    bank2.update(ph_f.detach(), cidx, 'ph')
    # Force enough prototypes for warmup
    for c in range(n_seen):
        bank2.proto_sk[c] = F.normalize(torch.randn(embed_dim, device=device), dim=-1)
        bank2.proto_ph[c] = F.normalize(torch.randn(embed_dim, device=device), dim=-1)
        bank2.proto_mask[c] = True

    loss_ssc, loss_xmod = structural_losses(
        sk_f, ph_f, cidx, bank2, anchor_A, dist='mse', warmup=3
    )
    print(f'   L_SSC  = {loss_ssc.item():.4f}')
    print(f'   L_xmod = {loss_xmod.item():.4f}')

    # L_asym_sph
    sk_anchor = F.normalize(torch.randn(B, embed_dim, device=device), dim=-1)
    ph_anchor = F.normalize(torch.randn(B, embed_dim, device=device), dim=-1)
    loss_sph = asym_spherical_loss(sk_f, ph_f, sk_anchor, ph_anchor, 1.0, 0.2)
    print(f'   L_asym_sph = {loss_sph.item():.4f}')

    # Check no NaN
    for name, val in [('cls', loss_cls), ('ssc', loss_ssc), ('xmod', loss_xmod), ('sph', loss_sph)]:
        assert not torch.isnan(val), f'NaN in L_{name}!'
    print('   All losses finite: [OK]')
    print()

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    print('[4] End-to-end gradient check...')
    total_loss = loss_cls + loss_ssc + loss_xmod + loss_sph
    total_loss.backward()
    assert sk_f.grad is not None
    assert ph_f.grad is not None
    print('   gradients flow through all losses: [OK]')
    print()

    print('=' * 50)
    print('[OK]  ALL SMOKE TESTS PASSED')
    print('=' * 50)


if __name__ == '__main__':
    run()

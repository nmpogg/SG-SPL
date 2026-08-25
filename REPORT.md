## 1. Nguyên nhân CLIP-AT baseline bị parameter leakage

CLIP-AT được thiết kế để chỉ fine-tune LayerNorm của visual encoder cùng với visual prompts. Tuy nhiên, logic đóng băng tham số chỉ xử lý các thuộc tính trực tiếp có tên `weight` và `bias`:

```python
for module in visual_encoder.modules():
    if not isinstance(module, nn.LayerNorm):
        if hasattr(module, "weight"):
            module.weight.requires_grad_(False)
        if hasattr(module, "bias"):
            module.bias.requires_grad_(False)
```

Cách làm này không khóa được các tham số có tên khác `weight` hoặc `bias`. Đặc biệt, `nn.MultiheadAttention` lưu QKV projection dưới tên `in_proj_weight` và `in_proj_bias`, nên hai tham số này vẫn có `requires_grad=True`. Các `nn.Parameter` được khai báo trực tiếp trong visual encoder cũng không bị khóa.

### Các tham số đang bị leak

- Multi-Head Attention:
  - `attn.in_proj_weight`
  - `attn.in_proj_bias`
- Visual embeddings:
  - `visual.class_embedding`
  - `visual.positional_embedding`
- Visual output projection:
  - `visual.proj`

Ngoài các tham số bị leak, mô hình vẫn fine-tune:

- Tất cả `LayerNorm.weight` và `LayerNorm.bias` trong visual encoder
- `sk_prompt`
- `img_prompt`

MLP, convolution và `attn.out_proj` vẫn bị đóng băng. `logit_scale` cũng không được fine-tune vì nó nằm ngoài visual encoder và không được đưa vào optimizer.

Trong thiết lập hiện tại, các tham số visual bị leak và LayerNorm được cập nhật với `lr_ln`; hai prompt được cập nhật với `lr_prompt`.

## 2. Mục tiêu thực nghiệm

Parameter leakage làm thay đổi đáng kể số lượng tham số được fine-tune. Vì vậy, cần đánh giá lại toàn bộ baseline và các thành phần loss dưới cùng một leaked freeze setup.

Mọi thí nghiệm cần giữ cố định dataset split, seed, CLIP backbone, số lượng prompt, optimizer, learning rate, batch size, số epoch và quy trình validation/test; chỉ thay đổi cấu hình loss đang khảo sát.

Ký hiệu:

- `$L_{tri}$`: triplet loss của CLIP-AT baseline
- `$L_{cls}$`: classification loss
- `$L_{SSC}$`: structural semantic consistency loss
- `$L_{xmod}$`: cross-modal structural loss
- `$L_{asym}$`: asymmetric spherical loss

## 3. Danh sách thí nghiệm

### 3.1. Thiết lập lại CLIP-AT baseline dưới leaked freeze

```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 1 --classification_weight=0.5 --ssc_weight 0 --xmod_weight 0 --sph_ph_weight 0 --sph_sk_weight 0 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```

### 3.2. Thêm L_SSC
```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 1 --classification_weight 0.5 --ssc_weight 1 --xmod_weight 0 --sph_ph_weight 0 --sph_sk_weight 0 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```
### 3.3. Thêm L_SSC, tắt L_tri

```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 0 --classification_weight 0.5 --ssc_weight 1 --xmod_weight 0 --sph_ph_weight 0 --sph_sk_weight 0 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```

### 3.4. Thêm L_SSC + L_xmod
```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 1 --classification_weight 0.5 --ssc_weight 1 --xmod_weight 1 --sph_ph_weight 0 --sph_sk_weight 0 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```

### 3.5. Thêm L_SSC + L_xmod, tắt L_tri
```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 0 --classification_weight 0.5 --ssc_weight 1 --xmod_weight 1 --sph_ph_weight 0 --sph_sk_weight 0 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```

### 3.6. Thêm L_asym
```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 1 --classification_weight 0.5 --ssc_weight 0 --xmod_weight 0 --sph_ph_weight 1 --sph_sk_weight 0.2 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```

### 3.7. Thêm L_asym, tắt L_tri
```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 0 --classification_weight 0.5 --ssc_weight 0 --xmod_weight 0 --sph_ph_weight 1 --sph_sk_weight 0.2 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```

### 3.8. Thêm L_SSC + L_xmod + L_asym (Full SG-SPL)
```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 1 --classification_weight 0.5 --ssc_weight 1 --xmod_weight 1 --sph_ph_weight 1 --sph_sk_weight 0.2 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```

### 3.9. Thêm L_SSC + L_xmod + L_asym, tắt L_tri
```bash
python experiments/train.py --dataset sketchy_2 --root <data_path> --n_prompts 3 --max_epochs 20 --triplet_weight 0 --classification_weight 0.5 --ssc_weight 1 --xmod_weight 1 --sph_ph_weight 1 --sph_sk_weight 0.2 --lr_ln 5e-4 --lr_prompt 1e-3 --batch_size 128 --independent_ln
```

## 4. Bảng so sánh

Kết quả đánh giá trên `sketchy_2`:

| Exp | $L_{tri}$ | $L_{cls}$ | $L_{SSC}$ | $L_{xmod}$ | $L_{asym}$ (`ph/sk`) | mAP@200 | P@200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3.1 | ✓ | ✓ | x | x | x | — | — |
| 3.2 | ✓ | ✓ | ✓ | x | x | — | — |
| 3.3 | x | ✓ | ✓ | x | x | — | — |
| 3.4 | ✓ | ✓ | ✓ | ✓ | x | — | — |
| 3.5 | x | ✓ | ✓ | ✓ | x | — | — |
| 3.6 | ✓ | ✓ | x | x | 1 / 0.2 | — | — |
| 3.7 | x | ✓ | x | x | 1 / 0.2 | — | — |
| 3.8 | ✓ | ✓ | ✓ | ✓ | 1 / 0.2 | — | — |
| 3.9 | x | ✓ | ✓ | ✓ | 1 / 0.2 | — | — |

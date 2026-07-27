# SG-SPL

## Chạy CLIP-AT baseline

Lệnh sau chạy CLIP-AT kiểu prompt tuning với chỉ `L_triplet` + `L_cls`, vô hiệu hoá các regularizer bổ sung (`L_SSC`, `L_xmod`):

```bash
python experiments/train.py \
  --dataset sketchy_2 \
  --root datasets/Sketchy/ \
  --l_ssc 0 \
  --l_x 0 \
  --l_sph_ph 0 \
  --l_sph_sk 0 \
  --batch_size 64 \
  --n_prompts 3 \
  --max_epochs 60 \
  --independent_ln
```
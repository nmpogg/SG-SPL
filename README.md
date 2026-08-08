# SG-SPL

## Chạy CLIP-AT baseline

Lệnh sau chạy CLIP-AT kiểu prompt tuning với chỉ `L_triplet` + `L_cls`, vô hiệu hoá các regularizer bổ sung:

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 1 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 \
  --xmod_weight 0 \
  --sph_ph_weight 0 \
  --sph_sk_weight 0 \
  --lr_ln 1e-3 \
  --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl \
  --ssc_temp 0.05
```

### Giải thích các tham số:

> **Lưu ý:** Các tham số hiện tại (`--n_prompts 1`, `--max_epochs 1`, `--triplet_weight 40.0`, `--classification_weight 0.5`, `--lr_ln 1e-3`, `--lr_prompt 1e-3`, `--batch_size 64`, `--independent_ln`) đang được thiết lập để cho kết quả cao nhất với baseline CLIP-AT (mAP@200 0.765, mô hình hội tụ sau 1 epoch). Các tham số còn lại là dành cho các hàm loss mới.

- `--dataset`: Tên phiên bản dataset sử dụng (ví dụ: `sketchy_2`).
- `--root`: Đường dẫn tới thư mục gốc chứa dữ liệu của dataset (chứa 2 thư mục con `sketch/` và `photo/`).
- `--n_prompts`: Số lượng visual prompt được thêm vào đầu vào của CLIP.
- `--max_epochs`: Số epoch huấn luyện tối đa. (Baseline hội tụ ở epoch 1).
- `--triplet_weight`: Trọng số cho Triplet Loss (`L_triplet`).
- `--classification_weight`: Trọng số cho Classification Loss (`L_cls`).
- `--ssc_weight`: Trọng số của hàm mất mát Semantic Structure Consistency (`L_SSC`). Đặt bằng `0` để tắt ở mô hình baseline.
- `--xmod_weight`: Trọng số của hàm mất mát Cross-modal Structure Consistency (`L_xmod`). Đặt bằng `0` để tắt.
- `--sph_ph_weight` / `--sph_sk_weight`: Trọng số Asymmetric Hyperspherical Anchoring của nhánh photo và sketch. Đặt bằng `0` để tắt.
- `--lr_ln`: Learning rate áp dụng cho các lớp LayerNorm của CLIP.
- `--lr_prompt`: Learning rate áp dụng riêng cho các prompt token mới được khởi tạo.
- `--batch_size`: Kích thước batch size mỗi bước huấn luyện.
- `--ssc_dist` / `--ssc_temp`: Cấu hình cho hàm tính khoảng cách và temperature của hàm `L_SSC`.
- `--independent_ln`: Học riêng LayerNorm của 2 nhánh sketch visual và photo visual. Nếu không dùng tham số này thì học chung LayerNorm trong visual encoder.

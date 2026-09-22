# SG-SPL

## Chạy NT-Xent baseline

Lệnh sau chạy prompt tuning với `L_NT-Xent + L_cls`, vô hiệu hoá các regularizer bổ sung:

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 0 \
  --classification_weight 1.0 \
  --ssc_weight 0 \
  --xmod_weight 0 \
  --sph_ph_weight 0 \
  --sph_sk_weight 0 \
  --nt_xent_weight 0.5 \
  --lr_ln 1e-6 \
  --lr_prompt 1e-5 \
  --batch_size 32 \
  --independent_ln \
  --ssc_dist kl \
  --ssc_temp 0.1
```

## Chạy SSC + xmod mới

Structural losses mặc định loại self-similarity trên đường chéo và có trọng số độc lập:

```bash
python experiments/train.py \
  --dataset sketchy_2 \
  --root /path/to/Sketchy \
  --n_prompts 1 \
  --triplet_weight 0 \
  --classification_weight 1.0 \
  --nt_xent_weight 0.5 \
  --ssc_weight 1.0 \
  --xmod_weight 1.0 \
  --ssc_dist kl \
  --ssc_temp 0.1 \
  --sph_ph_weight 0 \
  --sph_sk_weight 0 \
  --lr_ln 1e-6 \
  --lr_prompt 1e-5 \
  --batch_size 32 \
  --independent_ln
```

Dùng `--include_structural_diagonal` chỉ khi cần tái lập loss KL cũ. `--ssc_dist` hỗ trợ `mse`, `kl`, `sym_kl`, và `js`.

### Giải thích các tham số:

> **Lưu ý tương thích:** `L_cls` hiện lấy trung bình hai modality đúng theo công thức. Vì implementation cũ trả về tổng, `--classification_weight 1.0` mới tương đương về độ lớn với `--classification_weight 0.5` cũ.

- `--dataset`: Tên phiên bản dataset sử dụng (ví dụ: `sketchy_2`).
- `--root`: Đường dẫn tới thư mục gốc chứa dữ liệu của dataset (chứa 2 thư mục con `sketch/` và `photo/`).
- `--n_prompts`: Số lượng visual prompt được thêm vào đầu vào của CLIP.
- `--max_epochs`: Số epoch huấn luyện tối đa. (Baseline hội tụ ở epoch 1).
- `--triplet_weight`: Trọng số cho Triplet Loss (`L_triplet`).
- `--classification_weight`: Trọng số cho Classification Loss (`L_cls`).
- `--ssc_weight`: Trọng số của hàm mất mát Semantic Structure Consistency (`L_SSC`). Đặt bằng `0` để tắt ở mô hình baseline.
- `--xmod_weight`: Trọng số độc lập của Cross-modal Structure Consistency (`L_xmod`). Đặt bằng `0` để tắt.
- `--sph_ph_weight` / `--sph_sk_weight`: Trọng số Asymmetric Hyperspherical Anchoring của nhánh photo và sketch. Đặt bằng `0` để tắt.
- `--lr_ln`: Learning rate áp dụng cho các lớp LayerNorm của CLIP.
- `--lr_prompt`: Learning rate áp dụng riêng cho các prompt token mới được khởi tạo.
- `--batch_size`: Kích thước batch size mỗi bước huấn luyện.
- `--ssc_dist` / `--ssc_temp`: Khoảng cách (`mse`, `kl`, `sym_kl`, `js`) và temperature của structural losses.
- `--include_structural_diagonal`: Khôi phục self-similarity diagonal để ablation/tái lập loss cũ; mặc định diagonal bị loại.
- `--independent_ln`: Học riêng LayerNorm của 2 nhánh sketch visual và photo visual. Nếu không dùng tham số này thì học chung LayerNorm trong visual encoder.

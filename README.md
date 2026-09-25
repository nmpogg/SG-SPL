# SG-SPL

## Phương án A: prototype contrastive

Chạy `experiments/train.py` với dữ liệu và các tham số huấn luyện hiện có; loss mới bật mặc định với `--proto_weight 0.25 --proto_temp 0.07 --proto_reverse_weight 0`. `L_sph` đã được thay thế hoàn toàn trong bước huấn luyện. Prototype loss chỉ bắt đầu khi bank có đủ `--bank_warmup` lớp và chỉ dùng prototype từ các batch trước. Theo dõi `train_loss_proto` và `train_active_prototypes` trong log để kiểm tra thời điểm loss bắt đầu hoạt động.

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
  --proto_weight 0 \
  --nt_xent_weight 0 \
  --lr_ln 1e-3 \
  --lr_prompt 1e-3 \
  --batch_size 64 \
  --ssc_dist kl \
  --ssc_temp 0.05
```

### Giải thích các tham số:

> **Lưu ý:** Các tham số hiện tại (`--n_prompts 1`, `--max_epochs 1`, `--triplet_weight 40.0`, `--classification_weight 0.5`, `--lr_ln 1e-3`, `--lr_prompt 1e-3`, `--batch_size 64`) đang được thiết lập để cho kết quả cao nhất với baseline CLIP-AT (mAP@200 0.765, mô hình hội tụ sau 1 epoch). Hai nhánh sketch và photo luôn có visual encoder riêng. Các tham số còn lại là dành cho các hàm loss mới.

- `--dataset`: Tên phiên bản dataset sử dụng (ví dụ: `sketchy_2`).
- `--root`: Đường dẫn tới thư mục gốc chứa dữ liệu của dataset (chứa 2 thư mục con `sketch/` và `photo/`).
- `--n_prompts`: Số lượng visual prompt được thêm vào đầu vào của CLIP.
- `--max_epochs`: Số epoch huấn luyện tối đa. (Baseline hội tụ ở epoch 1).
- `--triplet_weight`: Trọng số cho Triplet Loss (`L_triplet`).
- `--classification_weight`: Trọng số cho Classification Loss (`L_cls`).
- `--ssc_weight`: Trọng số của hàm mất mát Semantic Structure Consistency (`L_SSC`). Đặt bằng `0` để tắt ở mô hình baseline.
- `--xmod_weight`: Trọng số của hàm mất mát Cross-modal Structure Consistency (`L_xmod`). Đặt bằng `0` để tắt.
- `--proto_weight`: Trọng số prototype contrastive loss, mặc định `0.25`. Đặt bằng `0` để tắt.
- `--proto_temp`: Temperature của cosine logits, mặc định `0.07`.
- `--proto_reverse_weight`: Trọng số chiều photo→sketch, mặc định `0` để chỉ dùng sketch→photo.
- `--bank_warmup`: Số lớp có prototype tối thiểu trước khi bật prototype loss; sample chưa có prototype được bỏ qua.
- `--lr_ln`: Learning rate áp dụng cho các lớp LayerNorm của CLIP.
- `--lr_prompt`: Learning rate áp dụng riêng cho các prompt token mới được khởi tạo.
- `--batch_size`: Kích thước batch size mỗi bước huấn luyện.
- `--ssc_dist` / `--ssc_temp`: Cấu hình cho hàm tính khoảng cách và temperature của hàm `L_SSC`.

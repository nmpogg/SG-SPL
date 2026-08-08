# SG-SPL — Kế hoạch Thí nghiệm Toàn diện

> **Baseline hiện tại:** CLIP-AT trên Sketchy split-2
> - Mô hình **hội tụ sau epoch 1** — các epoch sau không cải thiện thêm.
> Mọi thí nghiệm bên dưới dựa trên cấu hình baseline tốt nhất đã xác nhận (mAP@200 = 0.762).

---

## Quy ước chung

- **Dataset mặc định:** `sketchy_2` (Sketchy-Extended split-2, 104 seen / 21 unseen).
- **Metric:** mAP@200, P@200 (theo protocol chuẩn cho sketchy_2).
- **`--root`:** `/kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy` (thay đổi nếu cần).
- **Baseline args cố định** (trừ khi ghi rõ khác): `--n_prompts 1 --triplet_weight 30.0 --classification_weight 0.5 --lr_ln 1e-3 --lr_prompt 1e-3 --batch_size 64 --independent_ln --seed 42`.
- **`max_epochs`:** Dùng `3` cho các thí nghiệm chính thức (hoặc `1` để chạy nhanh kiểm tra).
- Mỗi thí nghiệm chỉ **thay đổi MỘT hoặc vài biến so với baseline** để cô lập tác động.

---

## Mục lục

1. [Pha 0: Baseline CLIP-AT](#pha-0-baseline-clip-at)
2. [Pha 1: Bật L_SSC (Semantic Structure Consistency)](#pha-1-bật-l_ssc)
3. [Pha 2: Bật L_xmod (Cross-modal Structure Consistency)](#pha-2-bật-l_xmod)
4. [Pha 3: Bật L_asym_sph (Asymmetric Hyperspherical Anchoring)](#pha-3-bật-l_asym_sph)
5. [Pha 4: Full SG-SPL — kết hợp tất cả](#pha-4-full-sg-spl)
6. [Pha 5: Ablation bảng "vàng" (cộng dồn)](#pha-5-ablation-bảng-vàng)
7. [Pha 6: Sweep hyperparameter bổ sung](#pha-6-sweep-hyperparameter)
8. [Pha 7: Multi-dataset benchmark](#pha-7-multi-dataset)
9. [Pha 8: Thí nghiệm nâng cao](#pha-8-thí-nghiệm-nâng-cao)

---

## Pha 0: Baseline CLIP-AT

> Mục tiêu: Xác nhận lại baseline, chạy trên nhiều cấu hình prompt và epoch để có mốc so sánh vững chắc.

### EXP-00: Baseline gốc (đã xác nhận) ✅

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 30.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Kết quả:** mAP@200 = **0.762** (hội tụ sau epoch 1). Đây là mốc so sánh cho MỌI thí nghiệm.

### EXP-01: Baseline — không có independent_ln

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 30.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --ssc_dist kl --ssc_temp 0.05
```
**Kết quả:** thấp hơn nhưng chưa log lại

### EXP-02: Baseline — n_prompts = 3 ✅

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 3 \
  --max_epochs 3 \
  --triplet_weight 30.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Kết quả:** mAP@200 = **0.754** (tụt so với n_prompts=1). Nhiều prompt = nhiều tham số → dễ overfit → cần regularizer để cải thiện.

### EXP-03: Baseline — n_prompts = 5

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 5 \
  --max_epochs 3 \
  --triplet_weight 30.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Kiểm tra xu hướng n_prompts cao hơn. Dựa trên EXP-02 (n_prompts=3 → 0.754), kỳ vọng tiếp tục tụt → bằng chứng cần regularizer.

### EXP-04: Baseline — triplet_weight = 40 ✅

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Kết quả:** mAP@200 = **0.765**, P@200 = **0.734**. Tốt hơn baseline 0.762 một chút.

### EXP-04b: Baseline — triplet_weight = 50 ✅

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 50.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Kết quả:** mAP@200 = **0.764**, P@200 = **0.734**. Không cải thiện thêm so với 40 → triplet_weight = 30–40 là vùng tối ưu.

### EXP-05: Baseline — chỉ có triplet, không cls

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Xác nhận đóng góp của `L_cls`. Nếu tụt nhiều so với EXP-00 → `L_cls` quan trọng, baseline của ta (có `L_cls`) là công bằng khi so với SpLIP/DRCPL.

---

## Pha 1: Bật L_SSC

> Mục tiêu: Ablation cho `L_SSC` — hàm loss ép cấu trúc quan hệ liên lớp (prototype) khớp với anchor text CLIP gốc.
> Giữ `L_xmod = 0`, `L_asym_sph = 0` để cô lập tác động của `L_SSC`.

### EXP-10: SSC với MSE, λ_ssc = 1.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Cấu hình gốc EBSeg dùng MSE. Đây là điểm khởi đầu trung thành với nguồn gốc.

### EXP-11: SSC với MSE, λ_ssc = 0.5

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0.5 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** SSC nhẹ hơn — liệu regularize ít hơn có tốt hơn?

### EXP-12: SSC với MSE, λ_ssc = 2.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 2.0 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** SSC mạnh hơn — xem có phản tác dụng (quá cứng) không.

### EXP-13: SSC với MSE, λ_ssc = 5.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 5.0 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Cận trên sweep λ_ssc MSE. Nếu tụt → quá nhiều regularization.

### EXP-14: SSC với MSE, λ_ssc = 0.1

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0.1 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Cận dưới sweep λ_ssc MSE.

### EXP-15: SSC với KL, T=0.05, λ_ssc = 1.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist kl --ssc_temp 0.05 \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** KL divergence thay MSE. T=0.05 (sắc) ép mạnh vào thứ hạng quan hệ gần nhất.

### EXP-16: SSC với KL, T=0.1, λ_ssc = 1.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist kl --ssc_temp 0.1 \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** T=0.1 (mặc định, mềm hơn 0.05). So sánh mức "sắc" của softmax khi tính KL.

### EXP-17: SSC với KL, T=0.2, λ_ssc = 1.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist kl --ssc_temp 0.2 \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** T=0.2 (mềm nhất) — ít tập trung vào top-1 relation, phân bố đều hơn.

### EXP-18: SSC với KL, T=0.05, λ_ssc = 0.5

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0.5 --ssc_dist kl --ssc_temp 0.05 \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```

### EXP-19: SSC với KL, T=0.05, λ_ssc = 2.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 2.0 --ssc_dist kl --ssc_temp 0.05 \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```

### EXP-1A: SSC — no_proto_grad (tắt gradient qua prototype batch)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --no_proto_grad \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Ablation gradient trick — khi bật `--no_proto_grad`, prototype batch không truyền gradient ngược. Kỳ vọng: **tệ hơn** bản có gradient (chứng minh gradient trick quan trọng).

### EXP-1B: SSC — EMA momentum = 0.5

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --ema_m 0.5 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** EMA momentum thấp → prototype thay đổi nhanh, phản ánh batch hiện tại nhiều hơn. Có thể nhiễu hơn.

### EXP-1C: SSC — EMA momentum = 0.99

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --ema_m 0.99 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** EMA momentum cao → prototype ổn định hơn nhưng phản ứng chậm. So sánh 0.5 vs 0.9 (default) vs 0.99.

### EXP-1D: SSC — bank_warmup = 5

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --bank_warmup 5 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Bật SSC sớm hơn (chỉ cần 5 class active thay vì 10). Xem SSC sớm có giúp hay gây nhiễu.

### EXP-1E: SSC — bank_warmup = 20

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --bank_warmup 20 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Bật SSC muộn hơn — để bank ổn định trước khi tính loss.

---

## Pha 2: Bật L_xmod

> Mục tiêu: Thử nghiệm L_xmod — ép cấu trúc liên lớp giữa sketch và photo giống nhau.
> Lấy **cấu hình SSC tốt nhất từ Pha 1** làm nền, rồi sweep λ_xmod.

### EXP-20: SSC(best) + xmod, λ_xmod = 0.5

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Thêm xmod ở mức vừa phải. Kỳ vọng: gain thêm trên nền SSC.

### EXP-21: SSC(best) + xmod, λ_xmod = 1.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 1.0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** xmod mạnh — ép hai modality khớp nhau chặt.

### EXP-22: SSC(best) + xmod, λ_xmod = 0.1

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.1 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** xmod nhẹ nhàng — constraint mềm.

### EXP-23: SSC(best) + xmod, λ_xmod = 2.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 2.0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** xmod rất mạnh — xem có phản tác dụng (hai modality sụp về cùng một embedding) không.

### EXP-24: Chỉ xmod KHÔNG CÓ SSC (control)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 \
  --xmod_weight 1.0 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist mse
```
**Mục đích:** Control — xmod một mình có tác dụng không? (Lưu ý: khi `ssc_weight=0`, xmod cũng bị vô hiệu vì nằm trong cùng nhóm `ssc_weight * (L_ssc + xmod_weight * L_xmod)` trong training step). **→ Cần kiểm tra lại code xem xmod có bị gắn với ssc_weight không.**

> ⚠️ **Chú ý:** Trong `model.py` dòng 220, tổng loss là `ssc_weight * (loss_ssc + xmod_weight * loss_xmod)`. Nếu `ssc_weight = 0` thì xmod cũng bằng 0. Để test xmod riêng, cần đặt `ssc_weight > 0` nhưng dùng trick: prototype bank vẫn cần hoạt động. Hoặc tách code.

---

## Pha 3: Bật L_asym_sph

> Mục tiêu: Ablation cho neo bất đối xứng — thí nghiệm "finding" chính.
> Tạm tắt SSC/xmod để cô lập tác động.

### EXP-30: Bất đối xứng — λ_ph = 1.0, λ_sk = 0.2 (default khuyến nghị)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Cấu hình bất đối xứng chuẩn — neo chặt photo (in-distribution), thả sketch (out-of-distribution). Đây là giả thuyết trung tâm.

### EXP-31: Đối xứng — λ_ph = 1.0, λ_sk = 1.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 1.0 --sph_sk_weight 1.0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Neo đối xứng (kiểu PromptSRC). Kỳ vọng: **tệ hơn** bất đối xứng vì ép sketch bám CLIP gốc = kìm thích nghi.

### EXP-32: Chỉ neo photo — λ_ph = 1.0, λ_sk = 0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 1.0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Thả sketch hoàn toàn. So sánh: liệu neo sketch nhẹ (0.2) có tốt hơn thả hoàn toàn (0) không?

### EXP-33: Bất đối xứng — λ_ph = 0.5, λ_sk = 0.1

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0.5 --sph_sk_weight 0.1 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Giảm cường độ neo cả hai — giữ tỷ lệ 5:1.

### EXP-34: Bất đối xứng — λ_ph = 2.0, λ_sk = 0.2

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 2.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Neo photo rất mạnh — xem có quá cứng không.

### EXP-35: Bất đối xứng — λ_ph = 1.0, λ_sk = 0.1

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.1 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Neo sketch nhẹ hơn nữa (0.1 vs 0.2).

### EXP-36: Bất đối xứng — λ_ph = 1.0, λ_sk = 0.3

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.3 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Neo sketch nặng hơn một chút (0.3 vs 0.2). Fine-grained sweep quanh default.

### EXP-37: Ngược — λ_ph = 0.2, λ_sk = 1.0 (sanity check)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 0.2 --sph_sk_weight 1.0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Sanity check — đảo ngược bất đối xứng (neo sketch chặt, thả photo). Kỳ vọng: **tệ** — xác nhận giả thuyết "photo in-distribution, sketch out-of-distribution".

---

## Pha 4: Full SG-SPL

> Mục tiêu: Kết hợp tất cả loss, dùng cấu hình tốt nhất từ Pha 1–3.

### EXP-40: Full SG-SPL — cấu hình mặc định

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Full SG-SPL với tất cả regularizer ở mức mặc định.

### EXP-41: Full SG-SPL — KL thay MSE

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist kl --ssc_temp 0.05 \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Full SG-SPL nhưng dùng KL divergence.

### EXP-42: Full SG-SPL — independent LN

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Full SG-SPL + independent LayerNorm cho mỗi modality. L_asym_sph có thể cộng hưởng tốt khi sketch branch có LN riêng.

### EXP-43: Full SG-SPL — n_prompts = 3

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 3 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Full SG-SPL + 3 prompts. Regularizer nên giúp nhiều hơn khi model phức tạp hơn (nhiều prompt = dễ overfit hơn).

### EXP-44: Full SG-SPL — n_prompts = 5

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 5 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Full + 5 prompts. So sánh: nếu baseline 5 prompts bị tụt (EXP-03) mà full SG-SPL 5 prompts tốt → regularizer cứu được.

---

## Pha 5: Ablation bảng "vàng" (cộng dồn)

> Bảng này sẽ đi vào paper — thêm từng thành phần một, xem gain cộng dồn.

| # | L_cls | L_SSC | L_xmod | L_asym_sph | EXP ID |
|---|-------|-------|--------|------------|--------|
| 1 | ✔     |       |        |            | EXP-01 |
| 2 | ✔     | ✔     |        |            | EXP-10 (best) |
| 3 | ✔     | ✔     | ✔      |            | EXP-20 (best) |
| 4 | ✔     |       |        | ✔          | EXP-30 |
| 5 | ✔     | ✔     | ✔      | ✔          | EXP-40 (best) |

### EXP-50: Ablation — full TRỪU SSC (giữ xmod + asym_sph)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 0 --xmod_weight 0 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --ssc_dist kl --ssc_temp 0.05
```
**Mục đích:** Chỉ neo bất đối xứng, không SSC/xmod. So sánh với full → tách đóng góp riêng của structural losses.

### EXP-51: Ablation — full TRỪ asym_sph (giữ SSC + xmod)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 0 --sph_sk_weight 0 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** SSC + xmod mà không neo → tách đóng góp của neo.

---

## Pha 6: Sweep hyperparameter bổ sung

> Các sweep nhỏ quanh cấu hình tốt nhất, fine-tune.

### EXP-60: lr_prompt = 5e-4 (giảm LR prompt)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 5e-4 \
  --batch_size 64 \
  --independent_ln
```

### EXP-61: lr_prompt = 5e-3 (tăng LR prompt)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 5e-3 \
  --batch_size 64 \
  --independent_ln
```

### EXP-62: lr_ln = 5e-4 (giảm LR LayerNorm)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 5e-4 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```

### EXP-63: lr_ln = 1e-4 (giảm LR LayerNorm rất thấp)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-4 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```

### EXP-64: triplet_margin = 0.2 (thay đổi margin)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --triplet_margin 0.2 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```

### EXP-65: triplet_margin = 0.5 (margin lớn hơn)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --triplet_margin 0.5 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```

### EXP-66: batch_size = 128

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 128 \
  --independent_ln
```
**Mục đích:** Batch lớn hơn → mỗi batch chứa nhiều class hơn → prototype bank cập nhật tốt hơn → SSC/xmod có thể tốt hơn.

### EXP-67: batch_size = 192

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 192 \
  --independent_ln
```
**Mục đích:** Batch size 192 — theo doc/idea.txt, đây là batch size gợi ý cho prototype bank.

### EXP-68: classification_weight = 1.0

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 1.0 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Tăng trọng số cls — phân loại mạnh hơn.

### EXP-69: classification_weight = 0.1

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.1 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Giảm cls — xem regularizer structural có đủ thay thế vai trò cls không.

---

## Pha 7: Multi-dataset benchmark

> Mục tiêu: Chạy cấu hình tốt nhất trên tất cả dataset + 3 seeds cho mean ± std.

### EXP-70: Sketchy split-1 (mAP@all, P@100)

```bash
!python experiments/train.py \
  --dataset sketchy_1 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Metric:** mAP@all, P@100 (100 seen / 25 unseen). Chạy thêm seed 0, 1, 2.

### EXP-71: TU-Berlin (mAP@all, P@100)

```bash
!python experiments/train.py \
  --dataset tuberlin \
  --root <path-to-tuberlin> \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Metric:** mAP@all, P@100 (220 seen / 30 unseen). Cần chuẩn bị dataset.

### EXP-72: QuickDraw (mAP@all, P@200)

```bash
!python experiments/train.py \
  --dataset quickdraw \
  --root <path-to-quickdraw> \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Metric:** mAP@all, P@200 (80 seen / 30 unseen). Cần chuẩn bị dataset.

### EXP-73–75: Chạy lại config tốt nhất với 3 seeds

Cho mỗi dataset, chạy thêm `--seed 0`, `--seed 1`, `--seed 2` → mean ± std.

---

## Pha 8: Thí nghiệm nâng cao

### EXP-80: Full SG-SPL + independent_ln + n_prompts 3

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 3 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Combo đầy đủ nhất — nếu regularizer hoạt động, cấu hình mạnh hơn (3 prompts + independent LN) nên tốt hơn.

### EXP-81: Full SG-SPL — precision fp32 (kiểm tra ổn định)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln \
  --precision 32
```
**Mục đích:** Kiểm tra xem mixed precision (16-mixed) có gây bất ổn cho structural losses không. Nếu fp32 tốt hơn nhiều → cần debug mixed precision.

### EXP-82: Full SG-SPL — KL + T=0.1 + xmod=1.0 (KL variant mạnh)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 40.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist kl --ssc_temp 0.1 \
  --xmod_weight 1.0 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```

### EXP-83: Full SG-SPL — triplet_weight = 10 (giảm triplet, tăng tương đối regularizer)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 10.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```
**Mục đích:** Giảm trọng số triplet → tỷ trọng regularizer tương đối cao hơn. Xem cân bằng nào tốt nhất.

### EXP-84: Full SG-SPL — triplet_weight = 50 (tăng triplet)

```bash
!python experiments/train.py \
  --dataset sketchy_2 \
  --root /kaggle/input/datasets/nmpogg/sketchy-yelamarthi/Sketchy/Sketchy \
  --n_prompts 1 \
  --max_epochs 3 \
  --triplet_weight 50.0 \
  --classification_weight 0.5 \
  --ssc_weight 1.0 --ssc_dist mse \
  --xmod_weight 0.5 \
  --sph_ph_weight 1.0 --sph_sk_weight 0.2 \
  --lr_ln 1e-3 --lr_prompt 1e-3 \
  --batch_size 64 \
  --independent_ln
```

---

## Tóm tắt: Tổng số thí nghiệm

| Pha | Mô tả | Số thí nghiệm |
|-----|--------|---------------|
| 0 | Baseline CLIP-AT | 6 |
| 1 | L_SSC ablation | 14 |
| 2 | L_xmod ablation | 5 |
| 3 | L_asym_sph ablation | 8 |
| 4 | Full SG-SPL | 5 |
| 5 | Ablation bảng vàng | 2 |
| 6 | Sweep hyperparameter | 10 |
| 7 | Multi-dataset | 6+ |
| 8 | Nâng cao | 5 |
| **Tổng** | | **~61 cấu hình** |

---

## Ghi chú quan trọng

> [!IMPORTANT]
> **Thứ tự chạy:** Luôn chạy theo thứ tự pha. Kết quả pha trước ảnh hưởng cấu hình pha sau.
> Ví dụ: cấu hình SSC tốt nhất từ Pha 1 sẽ được dùng trong EXP-20–23.

> [!WARNING]
> **Quy tắc một biến:** Khi so sánh, chỉ thay đổi MỘT tham số mỗi lần. Nếu cần thay nhiều, ghi rõ lý do.

> [!TIP]
> **Chiến lược tiết kiệm GPU:** Chạy `max_epochs 1` trước cho tất cả thí nghiệm để lọc nhanh. Chỉ chạy đầy đủ 20 epochs cho top-5 cấu hình hứa hẹn nhất.

> [!NOTE]
> **Trong code hiện tại:** `ssc_weight` nhân chung cả `L_SSC` và `xmod_weight * L_xmod` (xem `model.py` dòng 220).
> Do đó **không thể test xmod riêng khi `ssc_weight = 0`**. EXP-24 cần lưu ý điều này.

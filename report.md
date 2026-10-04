# Báo cáo Lab 16 - OCI (CPU + LightGBM)

## Báo cáo ngắn

1. **Môi trường:** OCI, region `ap-singapore-1`, compartment `ai-lab-compartment`, VCN `ai-lab-vcn`, VM `ai-cpu-node` shape `VM.Standard.E2.1.Micro` (1 vCPU / 1 GB RAM, thêm 2 GB swap); đo lúc 2026-10-04 08:19 UTC; bài làm ở nhánh `lab16-work`.
2. **Khác với README:** shape `VM.Standard.A1.Flex` báo "Out of capacity" nên dùng shape Always Free x86; tôi không tìm thấy ô cloud-init ở các màn hình đã mở (Security, Networking, Storage; chưa mở mục Advanced options của bước Basic information) nên cài thư viện thủ công bằng đúng các lệnh trong `cloud-init-cpu.yaml`; về SSH, tôi đặt cổng 22 chỉ cho IP `/32` đúng README (118.71.118.120) nhưng kết nối bị timeout vì IP công khai của nhà mạng thay đổi (đo được thêm 14.238.145.228 và 203.205.17.217), nên để chẩn đoán và hoàn thành lab tôi đã mở tạm `0.0.0.0/0`, xác thực chỉ bằng SSH key, và xóa luôn rule này khi xóa VCN sau khi nộp kết quả (lẽ ra nên dùng dải IP hẹp hơn hoặc Bastion/VPN, đây là điểm chưa đạt khuyến nghị bảo mật).
3. **Dữ liệu:** Credit Card Fraud, 284.807 dòng (~0,17% gian lận), chia train/validation/test = 170.883 / 56.962 / 56.962 (stratify, seed 42); early stopping và chọn hyperparameter chỉ dùng validation, test chỉ để đánh giá cuối.
4. **Kết quả (test):** AUC-ROC 0,9739 (tính từ xác suất `predict_proba`), Accuracy 0,9995, F1 0,8444, Precision 0,9268, Recall 0,7755, best iteration 98.
5. **Hiệu năng:** load data 3,58 s, training 11,42 s; inference 1 dòng 1,80 ms (trung bình 500 lần sau 50 lần warm-up); batch 1.000 dòng 0,0045 s (trung vị 20 lần) ≈ 221.068 dòng/giây; thời gian inference gồm chuyển DataFrame sang mảng và gọi `predict`.
6. **Nhận xét:** Accuracy cao chủ yếu do lớp âm chiếm đa số, nên F1/Recall phản ánh đúng hơn; cấu hình mặc định LightGBM sụp ở iteration 2 (validation AUC 0,9133), chọn `reg_lambda=1, min_child_samples=50` (validation AUC 0,9821, tốt nhất trong 5 cấu hình ở `hp_select.py`).
7. **Tài nguyên:** ảnh `top`/`free -h`/`ip -s link` chụp lúc máy rảnh (không phải lúc benchmark đang chạy): CPU ~99,8% idle, RAM dùng 376/954 MiB, swap 31 MiB; `ip -s link` là số tích lũy (RX ~550 MB gồm tải dataset và cài thư viện), không phải tốc độ.
8. **Chi phí:** ước tính ~$0 vì shape Always Free, boot volume 46,6 GB trong hạn mức miễn phí, không dùng NAT/Load Balancer; Cost Analysis (1-4/10/2026 UTC, nhóm theo Compartment, chụp 15:11 ngày 04/10/2026) hiển thị "no costs" nhưng Oracle trễ ~24 giờ nên Billing **chưa cập nhật tại thời điểm chụp**, cần kiểm tra lại sau.
9. **Dọn dẹp:** đã tải kết quả về laptop trước; instance Terminated lúc 15:29, VCN xóa lúc 15:35, boot volume Terminated lúc 15:38 (04/10/2026, giờ địa phương); token Kaggle đã thu hồi; bằng chứng ở `screenshots/5_` đến `8_`.

## Bảng số liệu

| Metric | Kết quả |
|---|---|
| Thời gian load data | 3,58 s |
| Thời gian training | 11,42 s |
| Best iteration | 98 |
| AUC-ROC | 0,9739 |
| Accuracy | 0,9995 |
| F1-Score | 0,8444 |
| Precision | 0,9268 |
| Recall | 0,7755 |
| Inference latency (1 row) | 1,80 ms |
| Inference throughput (1000 rows) | 0,0045 s ≈ 221.068 dòng/giây |

## Ghi chú bổ sung

- **Lý do dùng OCI:** billing account GCP của tôi ở trạng thái "not in good standing" và không mở lại được, nên tôi chuyển sang OCI theo `README_other_clouds.md`.
- **Chưa báo Lab Coach** việc OCI hết capacity và chuyển sang `E2.1.Micro` (shape này không phát sinh phí).
- **Nguyên nhân sụp ở iteration 2** của cấu hình mặc định chưa được kiểm chứng riêng; nhiều khả năng do lá có quá ít mẫu dương, thêm regularization thì ổn định.
- **Phần GPU + LLM (tùy chọn) không thực hiện.**
- **Tập ảnh:** `screenshots/1_benchmark_terminal.png`, `2_resources_top.png`, `3_resources_free_and_network.png`, `4_billing_cost_analysis_by_compartment.png`, `5_cleanup_instance_terminating.png`, `6_cleanup_instance_terminated.png`, `7_cleanup_vcn_deleted.png`, `8_cleanup_boot_volume_terminated.png`.

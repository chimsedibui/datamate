# Lịch sử điểm public và cách đọc

Public = điểm cổng thi trả về cho `submission.jsonl` của một lượt chạy trên 1000 câu public (14 shop `dev_001`-`dev_014`).
Số câu đúng / sai ước lượng từ điểm và số câu đã trả lời: đúng = (trả lời + 1000 × PA) / 2. Không có nhãn public: điểm chỉ dùng
ở mức tổng, không bao giờ làm nhãn cho từng câu.

| Ngày  | Bản (file nộp)                 | Thay đổi                                                                                                         | Trả lời | Đúng / sai (ước lượng) | Public    |
| ----- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ------- | ---------------------- | --------- |
| 11-03 | LoRA v1                        | SQL trên DuckDB, LoRA v1 (rejection sampling trên train)                                                         | 736     | 582 / 154              | 0.428     |
| 11-04 | v2 (`sql_v2_public_debtroute`) | LoRA v2, few-shot từ ngân hàng ví dụ train, macro công nợ, định tuyến câu nợ                                     | 821     | 656 / 165              | 0.491     |
| 11-04 | `sub_qty_unans`                | `qty` của view theo đơn vị cơ bản, trả lời `unanswerable` khi đa số                                              |         | 675 / 159              | 0.516     |
| 11-05 | `sub_v4_final`                 | Sinh mẫu mới, chuẩn hóa cách viết tên, hệ số đơn vị đọc từ giá                                                   | 837     | 719 / 118              | 0.60      |
| 11-05 | `sub_v6`                       | Mẫu thường bắt đầu bằng ghi chú ngắn trước SQL. Ví dụ macro công nợ trong ngân hàng                              | 875     | 752 / 122              | 0.63      |
| 11-05 | `sub_v7`                       | v6 + ẩn bảng thô (`--hide-raw`): bỏ                                                                              |         |                        | 0.623     |
| 11-06 | `sub_v8`                       | LoRA v3 (2910 dòng, có 26 câu do Qwen3.8-27B giải làm teacher, chỉ trên train). Kiểm tra view theo giờ và đơn vị | 889     | 764 / 125              | 0.639     |
| 11-06 | `sub_v9`                   | v8 + nhánh `sql-hints`, bỏ 2 phần chỉ khớp public. Lớp công nợ dev_003 soạn lại                                  | 902     | 792 / 110              | **0.683** |
| 11-11 | `public_v4`                    | LoRA v4 (thêm dữ liệu tổng hợp v1), cùng view đã lưu: số câu đúng không đổi, sai tăng                            | 914     | 793 / 121              | 0.672     |
| 11-11 | `pool_v3_v4`                   | Gộp phiếu mẫu của v3 và v4 (bỏ phiếu khi hai adapter không đồng ý)                                               | 898     | 793 / 105              | 0.688     |
| 11-12 | `public_fix1` (v3)             | Sửa theo kiểm toán public: 'kiểm tra lại' không phải trả hàng, mã đơn bỏ ngày ghép, hệ số đơn vị theo giá từng tháng, gợi ý cách nói (đh, bữa nay, cất hàng, vay), phẩm chất không đo được | 909     | 818 / 91               | 0.727     |
| 11-12 | `public_fix2` (v3)             | + tiền cấp đơn (khách trả, thu máy cũ, bớt) vào `v_orders`, gộp tên thợ viết kiểu 'a Đạt'                        | 906     | 830 / 76               | 0.754     |
| 11-12 | `pool_v3fix_v5`                | Gộp phiếu v3 + LoRA v5 (dữ liệu tổng hợp v2), cả hai trên pipeline đã sửa                                        | 883     | 828 / 55               | 0.773     |
| 11-13 | `pool_fix3`                    | + cột danh mục trong `v_sales` (size, số sữa, hãng, đơn vị cơ bản; `--catalog-cols`)                             | 887     | 834 / 53               | 0.781     |
| 11-13 | `public_fresh_v3`              | Đường private: v3 trên view soạn mới v11 (dev_013 không có view, dev_004 nhân sai hệ số)                         | 894     | 791 / 103              | 0.688     |
| 11-13 | `public_fresh3_v3`             | Đường private: v3 trên view soạn mới v13 (kiểm tra soạn view mới, quy đổi lại từ số lượng gốc)                   | 911     | 842 / 69               | 0.773     |
| 11-13 | **`pool_fresh3`**              | Đúng như bản Docker: view soạn mới v13, lượt v3 + lượt v5, gộp phiếu                                             | 896     | 848 / 48               | **0.800** |

## Đọc các bước tăng

- **v6 → v8 (+0.9 điểm):** LoRA v3 gần như không đổi hành vi trên public: 105 câu khác v6, ròng khoảng +12 đúng / +3 sai. Dữ liệu
  SFT v3 phần lớn giống v2, và phần khó của public (4 ngành không có trong train, từ địa phương, viết tắt tiền) nằm ngoài train.
- **v8 → v9 (+4.4 điểm):** chỉ khoảng 90 câu khác v8 nhưng ròng +28 đúng / -15 sai. Phần lớn đến từ gợi ý theo dạng câu hỏi (mỗi gợi ý
  là một luật trong rulebook, đã đối chiếu với đáp án train) và lớp công nợ mới của dev_003 (lớp cũ báo 26.288 đơn còn nợ).
- **v9 so với 0.696 (-1.3 điểm):** hai phần bỏ có chủ đích (danh sách từ đồng nghĩa lấy từ tên món public, gợi ý dòng sản phẩm. Hai phần này chạm
  khoảng 10 câu public, không kiểm chứng được trên train), LoRA v3 thay v2, view khác, và dao động giữa các lần chạy.

## Holdout không phải cổng

`splits/holdout600.txt` (632 câu train của 6 shop) đo cao hơn public một khoảng lớn và không đều: LoRA v3 đã học 724 dòng của
chính các câu này, và các shop public có 4 ngành không có trong train (210/1000 câu). Dùng holdout như smoke test (code chạy,
không tụt mạnh). Thay đổi thật được đánh giá bằng lượt nộp public, so cùng câu với bản trước bằng các cờ không cần nhãn
(`analysis/flags.py`, `notebooks/03_evaluation.ipynb`).

## Rủi ro đang mở

- Tới 11-12, mọi điểm public dùng view và lớp công nợ đã lưu từ các lượt trước (`--views-from`). Từ 11-13, các dòng `fresh` soạn lại mọi lớp như bản Docker; `pool_fresh3` (0.800) là cấu hình sẽ nộp, gần với điểm private nhất.
- Từ 11-11 các bản sửa đến từ kiểm toán: tự tính lại câu trả lời public từ file gốc theo README và rulebook (không có nhãn), tìm lỗi chung rồi sửa ở lớp dữ liệu hoặc gợi ý. Không có câu hỏi hay dữ liệu test nào đi vào dữ liệu train.
- Mọi điểm public ở trên (tới 11-12) dùng view và lớp công nợ đã lưu từ các lượt trước (`--views-from`). Bản Docker cho private soạn lại
  tất cả: `FRESH=1 bash scripts/run_public.sh` đo điểm khi soạn mới.
- View của dev_014 từng đếm 2292-2299 khách (mã khách là tên + số điện thoại viết nhiều kiểu). Từ 11-13 `views.build` gộp mã khách dạng tên bỏ qua hoa/thường, dấu, khoảng trắng: 987 khách, khớp kiểm toán.
- Câu về tồn kho theo lô (khoảng 31 câu public, chủ yếu dev_006) chưa có lớp riêng.

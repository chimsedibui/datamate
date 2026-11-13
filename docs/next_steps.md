# DataMate: trạng thái cải tiến và kế hoạch test

Ý tưởng lấy từ NVIDIA auto-ontology. Số liệu đo trên mẫu `--per-biz 5` của tập train (150 câu, cùng seed).

## Đã triển khai

| Mục | Code | Đã kiểm chứng |
|---|---|---|
| Answerability gate: đáp án 0/rỗng, tỷ trọng ngoài 0–100%, business chỉ có raw → abstain | `checks.gate`, `checks.gate_source` | Chạy thật với LLM: PA 0.220 → 0.273 |
| Value validation + tự sửa literal | `checks.value_checks`, `checks.auto_fix` | Chạy thật với LLM |
| Vòng tự sửa (`max_repairs=2`) | `agent.Agent.answer` | Chạy thật với LLM |
| Sanitize output mapping | `validate.sanitize` | Offline: hết lỗi trùng/thừa cột; chưa build lại |
| Làm tròn tiền ≥ 1000 | `agent._money` | Chạy thật: +3 câu (0.407 → 0.447) |
| Khách vãng lai (`kl`, `khách lẻ`, `<NA>`) → null; bỏ nhóm khách của đơn vãng lai | `validate.null_walk_in` (lúc build và lúc load) | Offline |
| `purchases.amount` lưu nhầm tổng phiếu → tính lại = qty × unit_cost | `validate.repair_tables`; check trong `validate` | Offline: phát hiện ở biz_123, biz_128 |
| Check số phần tử ("ghi 3 số", "top 5"); có null/khách lẻ trong xếp hạng; ép table sang list | `checks.expected_count`, `checks.coerce`, `review` | Offline |
| Helper quy đổi đơn vị `qty_in`, `avg_price` | `helpers.py` | Offline: 5/5 câu khớp đáp án chuẩn |
| Helper xác định khách `with_customer` (theo mã; không có mã thì theo tên, bỏ khác cách viết và số điện thoại) | `helpers.py` | Offline |
| Helper công nợ `debts`, `debt_flow` (rulebook §4) | `helpers.py` | Offline: 106/113 câu nợ đúng; 7 câu còn lại do harness parse sai |
| Gợi ý helper theo ý định câu hỏi (nợ, đơn vị, giá một X, xếp hạng khách) | `qparse.helper_hints` | Lý do: khi LLM tự viết code thì đúng 24/77, khi dùng helper thì đúng 38/52 |
| 4.1 Glossary từ viết tắt (dthu, bn, hqua, bill…) | `qparse.normalize_question` | Offline |
| 4.2 Domain summary lúc build (thêm 1 lời gọi LLM mỗi business; đưa đơn vị vào prompt QA) | `Mapper.domain_summary`, `agent.describe_tables` | Chưa chạy (cần build) |
| 4.3 Phản hồi lỗi giá trị yêu cầu "chỉ sửa literal"; ghi log `skeleton_changed` | `checks.skeleton`, `agent` | Offline bằng LLM giả |
| 4.4 Cache findings theo business | đã gỡ: không lần nào được dùng trên 4000 câu dev/holdout/public, và làm kết quả phụ thuộc thứ tự luồng | — |
| 4.5 Luật cho file archive/backup; luật `purchases.amount` là tiền dòng | `mapper.RULES` 20–21 | — |

Không làm phần trích synonym từ README: không README nào trong train/public_test định nghĩa từ viết tắt.

Tắt bằng env: `DM_GATE=0` (tắt toàn bộ gate), `DM_ABSTAIN_RAW=0` (vẫn trả lời từ raw).

## Vòng 3: giúp Qwen3-14B tự build semantic layer (đã triển khai)

Lý do: khi BTC chấm, layer do **Qwen3-14B** build, không phải gpt-4o. Log cũ cho thấy Qwen chỉ build được 0/13 business khi tắt thinking và 3/13 khi bật thinking (log có nhiễu do giới hạn request của model free).

| Mục | Code |
|---|---|
| Loader mẫu: 21 loader đã qua validate từ train, chọn 2 mẫu theo độ trùng tên cột + cùng ngành, đưa vào prompt build | `loader_examples.py`, `scripts/release/build_loader_library.py`; nguồn gốc nằm trong `training_code/loader_examples/` |
| Chấp nhận layer chỉ còn cảnh báo chất lượng dữ liệu, khi core tables đúng cấu trúc | `mapper._accept_with_warnings` |
| Mỗi business tối đa 25 phút, 12k token thinking mỗi lần thử, chỉ gửi lại lần thử gần nhất | `Mapper.build`, `predict.sh` |
| Trả góp (`installment`) tính là nợ, theo rulebook public | `helpers._CREDIT_METHODS` |

## Kế hoạch test

1. **Đo Qwen build trên OpenRouter** (khoảng $2, không cần VM):
   ```bash
   bash scripts/dev/measure_qwen_build.sh
   ```
   Ra 2 con số: số layer build OK trên 30 business (gpt-4o build được 25/30) và PA trên mẫu 150 câu (q14e: 0.447). Muốn so A/B không dùng loader mẫu thì chạy thêm `EXAMPLES=0 bash scripts/dev/measure_qwen_build.sh`.
2. **Test image trên VM GCP** (`scripts/gcp/README.md`), chỉ chạy khi bước 1 cho kết quả chấp nhận được.

**Lưu ý chi phí:** `LLM_BUDGET_USD` là trần cho mỗi process, tính theo `LLM_PRICE_IN/OUT`. Script ở bước 1 đã đặt giá Qwen ($0.12/$0.24 mỗi 1M token).

## Câu cần xem trong lần test tới
- `skeleton_changed` trong log: nếu LLM viết lại cả code khi chỉ được yêu cầu sửa giá trị, và kết quả thường sai, thì thêm gate cho trường hợp này.
- Tỷ lệ code có gọi `debts` / `qty_in` sau khi đã thêm gợi ý.
- Gate raw: hiện chỉ hơi có lợi (18 đúng / 25 sai). Nếu build lại được thêm business thì gate này ít khi phải dùng.

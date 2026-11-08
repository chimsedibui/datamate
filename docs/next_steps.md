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
| 4.4 Cache findings theo business: literal đã sửa được dùng lại cho câu sau, không tốn lời gọi | `checks.learned_values`, `Agent._findings` | Offline bằng LLM giả |
| 4.5 Luật cho file archive/backup; luật `purchases.amount` là tiền dòng | `mapper.RULES` 20–21 | — |

Không làm phần trích synonym từ README: không README nào trong train/public_test định nghĩa từ viết tắt.

Tắt bằng env: `DM_GATE=0` (tắt toàn bộ gate), `DM_ABSTAIN_RAW=0` (vẫn trả lời từ raw).

## Kế hoạch test (một lần, khi có key mới)

1. **Mẫu chuẩn**, so với q14e (PA 0.447):
   ```bash
   python src/predict.py --split data/train --work work/train --out out/train_v2 --per-biz 5 --jobs 6
   python src/evaluate.py --pred out/train_v2/submission.jsonl --split data/train
   ```
2. **Tập công nợ + đơn vị** (270 câu): file câu hỏi nằm trong scratchpad của phiên này. Có thể tạo lại bằng cách lọc `questions.jsonl` theo các từ: nợ, mua chịu, kg, m3, lon, thùng, "một <đơn vị>".
3. **Build lại 5 business lỗi** để kiểm chứng sanitize và domain summary:
   ```bash
   python src/build_semantic.py ... --retry-failed
   ```

**Lưu ý chi phí:** `LLM_BUDGET_USD` là trần cho mỗi process, tính theo `LLM_PRICE_IN/OUT` (đang đặt giá gpt-4o là 2.5/10). Khi chạy model khác, cần đặt giá tương ứng, nếu không trần sẽ cắt sớm. Lần chạy tập 270 câu bị cắt ở câu thứ 140 vì lý do này.

## Câu cần xem trong lần test tới
- `skeleton_changed` trong log: nếu LLM viết lại cả code khi chỉ được yêu cầu sửa giá trị, và kết quả thường sai, thì thêm gate cho trường hợp này.
- Tỷ lệ code có gọi `debts` / `qty_in` sau khi đã thêm gợi ý.
- Gate raw: hiện chỉ hơi có lợi (18 đúng / 25 sai). Nếu build lại được thêm business thì gate này ít khi phải dùng.

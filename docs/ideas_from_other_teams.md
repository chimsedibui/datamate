# Ý tưởng từ lời giải của các đội khác (đã sàng lọc)

Nguồn: 10 repo công khai của Track 2 DataMate, đọc ngày 13/11/2026. Đây là repo đội 0.800 (`hugebenevolence`, đã lưu ở nhánh
`ref/hugebenevolence-datamate`) và 9 repo trong danh sách. Chỉ giữ cơ chế **tổng quát**: không lấy gì gắn với mã shop,
tên món hay đáp án public. Số "đo trên log" là kết quả phát lại cơ chế đó trên log của mình (dev_v8, holdout_v8 có nhãn;
public_v10 không nhãn).

## Bức tranh chung

| Repo | Cách làm | Điểm công bố | Đáng học |
|---|---|---|---|
| hugebenevolence | Qwen3.5-9B + LoRA viết SQL DuckDB trên view chuẩn hoá, 8 mẫu + vote, gộp phiếu 2 adapter | **public 0.800** (848 đúng / 48 sai) | vote + abstain, đa dạng mô hình |
| nhanlenhanle | 2 phương pháp (kế hoạch JSON + SQL), mỗi cái SC ở T 0/0.5/0.9, gộp lại | holdout tự chia 0.837 | gộp hai phương pháp, các check câu hỏi ↔ đáp án |
| phamchivy | Qwen3-14B agent nhiều lượt, 4 kiểu giải, vote có trọng số | public 0.380 → 0.521 | đo được: đáp án 0 chỉ do LLM đưa ra thì ròng −17 trên public |
| VanAnh-13 | Qwen3-14B code-act, verifier, vote, logistic gate | không có | làm tròn half-up, prompt khác nhau cho từng mẫu |
| Kyoyooo | DuckDB + QuestionFrame + gate | chưa chạy với model | xếp hạng làm bằng code, độ phủ kỳ tính bằng code |
| thaibinh0203, HoL0ngg | SQL trên PostgreSQL / DuckDB, model 4–12B | PA âm trên train | khuôn SQL cho từng kiểu câu phần trăm |
| StewedSalmon, levietbao1506, Oxazite | luật thuần / khung sườn / chỉ có README | không có | không có gì dùng được |

Kết luận chính: so với đội 0.800, mình **đúng nhiều câu hơn** (khoảng 858 so với 848) nhưng **sai nhiều hơn hẳn**
(khoảng 110 so với 48). Khoảng cách nằm ở chuyện biết lúc nào nên bỏ qua, không nằm ở khả năng giải.

## Ý tưởng nổi bật, xếp theo lợi ích ước tính

### 1. Vote nhiều mẫu và abstain khi các mẫu không đồng ý (đã làm: v11)
- Nguồn: hugebenevolence (`voting.py`), nhanlenhanle (`hybrid.py`), VanAnh-13 (`solver.py: vote`).
- Bỏ phiếu theo đúng cách scorer so sánh đáp án. Chỉ trả lời khi ≥ 50% mẫu cùng đáp án, không thì abstain.
- Với PA, một câu sai bị đổi thành abstain là +1 điểm.
- Ở mình: nhánh `adaptive-vote` / `v11`. K=3, dừng sớm khi 2 mẫu đầu đã giống nhau, khoá vote theo đúng `scoring.py`.

### 2. Đa dạng mẫu bằng prompt, không chỉ bằng temperature (đã làm: v12)
- Nguồn: VanAnh-13 (`agent/prompts.py: VARIANTS`), hugebenevolence (2 system prompt × K/2).
- Vote chỉ có ích khi các mẫu sai **khác nhau**. Mẫu chỉ khác temperature thường sai giống nhau.
- Ở mình: `SC_VARIANTS` trong `qa/agent.py`. Mẫu 1 phải viết kế hoạch dạng comment trước khi code (bảng, giá trị lọc,
  kỳ, luật trạng thái, phép gộp). Mẫu 2 phải chốt đơn vị của đáp án, phạm vi dòng và cách xử lý đồng hạng.
- Ghi chú được đặt **cuối** user message, nên system prompt và bảng mô tả vẫn là prefix chung, cache phía server
  vẫn dùng được. Mẫu 0 giữ nguyên prompt cũ.
- Hugebenevolence báo "ghi chú trước SQL" giúp khoảng +3 điểm public (0.60 → 0.63).

### 3. Phương pháp thứ hai trong vote (chưa làm: lợi ích lớn nhất còn lại)
- Nguồn: nhanlenhanle `hybrid.py: both()`. Nhóm này đo được: gộp 2 phương pháp 0.837, chỉ SQL 0.791, chỉ JSON 0.693.
- Một mẫu giải bằng SQL (DuckDB trên chính các frame chuẩn), các mẫu còn lại bằng pandas. Chỉ trả lời khi hai phương
  pháp đồng ý. Một bên trả lời còn bên kia báo "không có dữ liệu" thì abstain.
- Công sức trung bình: cần prompt SQL riêng và phải viết lại các helper (nợ, quy đổi đơn vị) dưới dạng macro.

### 4. Abstain có hiệu chỉnh (chưa làm)
- Nguồn: VanAnh-13 (`agent/confidence.py`, `eval/calibrate.py`).
- Logistic regression trên các đặc trưng: tỉ lệ phiếu, số đáp án khác nhau, số vòng review, số lỗi khi chạy code, kiểu
  đáp án có khớp kiểu dự đoán từ câu hỏi không, có phải unanswerable không, kỳ có nằm ngoài dữ liệu không.
- Fit trên shop train để riêng. Ngưỡng chọn để tối đa PA, cộng thêm biên an toàn cho các ngành chưa gặp.
- Cần log SC có nhãn (lượt `out/dev_sc` là bước đầu).

### 5. Check câu hỏi ↔ đáp án bằng luật cố định, sai thì cho viết lại (phần lớn đã có)
- Nguồn: nhanlenhanle `_sql_mismatch` / `_mismatch`, phamchivy `sanity_warnings`.
- Mình đã có trong `checks.review`: đếm không ra số nguyên, sai số phần tử top-k, đáp án 0 hoặc rỗng, tỉ trọng ngoài
  0–100%, null hoặc khách lẻ trong xếp hạng, unanswerable mâu thuẫn với khoảng dữ liệu.
- Còn thiếu và có thể thêm:
  - câu có/không phải trả boolean;
  - "ngày nào" phải trả date;
  - đếm số dòng thay vì `nunique` số đơn;
  - ngưỡng tiền của đơn lại áp lên tiền từng dòng;
  - hỏi một tháng mà code tính cả năm.
- Đo trên public_v10: chỉ 1 câu có/không trả sai kiểu. Lợi ích nhỏ.

### 6. Làm tròn half-up (đã làm: v12)
- Nguồn: VanAnh-13 (`core/numbers.py: round_half_up`), thaibinh0203.
- Rulebook ghi "nửa đồng làm tròn lên", trong khi `round()` của Python làm tròn kiểu ngân hàng: `round(1234.5) == 1234`.
- Ở mình: `parsers.round_half_up`, dùng cho mọi chỗ làm tròn đáp án. Tiền được làm tròn từ giá trị gốc, không làm tròn
  2 lần.
- Đo trên log: không đáp án nào đổi. Đúng luật nhưng hiếm gặp.
- Phát hiện kèm: 1 câu holdout model tự dùng `np.ceil`, vì hiểu nhầm "làm tròn lên" là luôn làm tròn lên. Hiện chưa sửa,
  vì phải sửa system prompt và sẽ mất cache.

### 7. Kỳ nằm hoàn toàn ngoài dữ liệu thì unanswerable (đã làm: v12)
- Nguồn: rulebook §7 ("kỳ nằm hoàn toàn ngoài khoảng ngày có dữ liệu … không trả lời được, không phải 0"), VanAnh-13
  `verify.py`, Kyoyooo `temporal/resolve.py: coverage`.
- Ở mình: `checks.periods_outside`. Chỉ đổi những câu **đã bị gate chặn** (0 hoặc rỗng) thành `unanswerable`, không
  bao giờ đè lên một đáp án đã có.
- Đo trên log: +1 câu holdout (đang abstain, đáp án đúng là unanswerable). Public: 0 câu.

### 8. Bẫy unanswerable (đã có, nhắc lại)
- nhanlenhanle: 28/30 câu unanswerable đưa ra vì template thất bại bị chấm sai trên public. phamchivy: đáp án 0 chỉ do LLM
  đưa ra thì ròng −17.
- Bài học: chỉ trả unanswerable khi có bằng chứng tính bằng code (thiếu bảng hoặc cột, kỳ nằm ngoài dữ liệu).
- Mình đã làm đúng tinh thần này: `gate_tables`, và luật 7 ở trên.

## Đã thử và loại bỏ

- **Kiểm tra tên ở đáp án có trong data** (VanAnh-13 `verify.py`): đo trên log, dev và holdout không bắt câu nào;
  public bắt 2 câu, đều là "bằng nhau" (đáp án hợp lệ). Code của mình vốn lấy tên từ dataframe nên không có lỗi gõ lại
  tên.
- **Coi đồng hạng ở mốc k là lỗi** (VanAnh-13, Kyoyooo): giả định "BTC đã loại câu có đồng hạng" mâu thuẫn với rulebook
  §8, vốn định nghĩa cách phá đồng hạng (theo mã tăng dần, không có mã thì theo tên). Nên làm ngược lại: kiểm tra code
  có phá đồng hạng đúng luật không.
- **Kiểm tra phần trăm ≤ 1.5 hoặc ≥ 100**: 10 câu public có phần trăm ≤ 1.5, đều là giá trị hợp lý (tỉ lệ huỷ khoảng 1%,
  tăng trưởng âm).
- **Sửa lỗi gõ Telex** (VanAnh-13, nhanlenhanle): 128 câu train nhưng chỉ khoảng 3 câu public.

## Việc nên làm tiếp (sau v12)

1. Mẫu SQL làm phương pháp thứ hai trong vote (mục 3).
2. Abstain có hiệu chỉnh, fit trên log SC có nhãn (mục 4).
3. Benchmark từ shop train bị biến đổi (đổi tên cột, nhãn trạng thái lạ, dòng tổng; nhanlenhanle `stress/`,
   phamchivy `make_dirty.py`), vì điểm dev và holdout hiện không dự báo được điểm public.
4. Đưa vào prompt luật phá đồng hạng (rulebook §8) dưới dạng helper xếp hạng: sắp theo tiêu chí, rồi theo mã tăng dần.

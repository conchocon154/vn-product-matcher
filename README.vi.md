# Vietnamese Product Matcher

Khớp tên hàng tiếng Việt gõ tự do về đúng mã SKU trong catalogue.

> Bản tiếng Anh: [README.md](README.md) — số liệu và phân tích đầy đủ.

## Bài toán

Nhân viên cửa hàng gõ `oc lgn 10x50 i304`. Catalogue ghi mặt hàng đó là
**Bu lông lục giác ngoài 10x50x17 inox 304**. Hai chuỗi gần như không có gì
chung: `ốc` và `bu lông` là hai từ khác nhau chỉ cùng một con bu lông, `lgn` là
viết tắt, `i304` là mác thép, còn cỡ chìa `17` thì người dùng không gõ.

Dự án xây và **đo** bộ khớp tên cho bài toán đó trên catalogue kim khí điện nước
thật, 1.827 mặt hàng.

## Kết quả

Tìm kiếm trên toàn bộ 1.827 SKU. Tập `unseen` gồm truy vấn của 274 SKU **bị loại
hoàn toàn khỏi quá trình huấn luyện**, dùng để kiểm tra model có thực sự học
cách đặt tên hàng tiếng Việt hay chỉ học thuộc catalogue.

| Phương pháp | R@1 (seen) | R@1 (unseen) | MRR (unseen) |
|---|---|---|---|
| Khớp chuỗi chính xác | 8,4% | 8,3% | 0,083 |
| BM25 (SQLite FTS5) | 82,2% | 83,4% | 0,853 |
| RapidFuzz | 89,6% | 88,2% | 0,920 |
| TF-IDF n-gram ký tự | 94,8% | 95,0% | 0,970 |
| Encoder chưa fine-tune | 65,0% | 63,8% | 0,721 |
| **Encoder đã fine-tune** | **97,1%** | **96,9%** | **0,980** |
| Encoder + rerank kích thước | 96,9% | 96,8% | 0,980 |

Hai cột `seen` và `unseen` chênh nhau 0,2 điểm, nghĩa là model khớp được cả
những SKU chưa từng thấy.

## Ba kết luận nói thẳng

**1. Fine-tune mới là thứ quyết định, không phải chọn model nào.**
Dùng nguyên bản, `paraphrase-multilingual-MiniLM-L12-v2` chỉ đạt 63,8% — *thua
xa* baseline TF-IDF viết bằng sáu dòng scikit-learn. Cũng bộ trọng số đó sau khi
fine-tune đạt 96,9%. Nếu bê thẳng một embedding model vào mà không thích ứng với
lĩnh vực, hệ thống sẽ **tệ đi** chứ không tốt lên.

**2. Khoảng cách so với baseline từ vựng là thật, nhưng khiêm tốn.**
TF-IDF n-gram ký tự đạt 95,0%. Encoder fine-tune hơn 1,9 điểm, kiểm định McNemar
p = 1,22e-03, khoảng tin cậy 95% của khoảng cách là [+0,80%, +2,94%]. Có ý nghĩa
thống kê — và đúng 1,9 điểm, không phải mức nhảy vọt mà cụm từ "áp dụng deep
learning" thường gợi ra. Chỗ encoder thật sự đáng giá là nhóm truy vấn dùng từ
đồng nghĩa: 93,6% → 97,4%.

**3. Ý tưởng hybrid rerank của tôi là công cốc, và phép đo nói ra điều đó.**
Thiết kế ban đầu giả định encoder xử lý kém con số, nên thêm một lớp rerank chấm
điểm trùng khớp kích thước và vật liệu. Kết quả: không thay đổi gì
(96,9% → 96,8%, McNemar p = 1,00, khoảng tin cậy [-0,49%, +0,37%]). Fine-tune đã
dạy encoder đọc được `10x50x17` rồi. Lớp rerank vẫn nằm trong `retrievers.py`
như một kết quả âm có ghi chép, nhưng API phục vụ encoder thuần: hybrid tốn
khoảng 4 lần độ trễ mà không đổi lại được độ chính xác nào đo được.

## Hạn chế

Đọc phần này trước khi tin vào các con số.

- **Truy vấn là dữ liệu sinh, không phải log thật.** Cửa hàng không lưu lại
  những gì nhân viên đã gõ, nên mọi truy vấn ở đây được tạo bằng cách làm nhiễu
  tên trong catalogue. Các kiểu nhiễu chọn theo cách tên hàng thực sự biến dạng
  trên phiếu đặt hàng viết tay, nhưng đó vẫn là **mô hình hóa** hành vi người
  dùng chứ không phải mẫu quan sát được.
- **Kết quả về từ đồng nghĩa có phần vòng tròn.** Truy vấn huấn luyện và truy vấn
  kiểm thử lấy từ đồng nghĩa từ cùng một từ điển, nên encoder được thưởng vì học
  đúng ánh xạ mà bộ sinh dữ liệu vốn đã biết. Với từ đồng nghĩa nằm ngoài danh
  sách đó, kỳ vọng kết quả gần mức TF-IDF hơn là 97,4%.
- **Đã bỏ cột giá.** Nguồn dữ liệu là bảng giá nhà cung cấp; repo chỉ công bố
  tên hàng, quy cách và đơn vị tính.
- **Một catalogue, một ngành hàng.** 1.827 mặt hàng kim khí điện nước, chưa thử
  trên lĩnh vực nào khác.

## Điểm kỹ thuật đáng chú ý

**Sinh truy vấn chia hai tầng.** Nhiễu chia làm hai loại, trộn lẫn là hỏng nhãn:
nhiễu *bề mặt* (lỗi gõ, mất dấu, viết tắt, từ đồng nghĩa) giữ nguyên thông tin;
nhiễu *mất thông tin* (bỏ từ, cắt `10x50x17` còn `10x50`, lược hết tính từ) có
thể tạo ra truy vấn khớp nhiều SKU. Ví dụ `tán xd sắt xi xám` khớp cả loại M12
lẫn M22 — chấm điểm nó với một đáp án duy nhất là phạt oan bộ khớp. Nên nhiễu
mất thông tin chạy trước, kết quả được kiểm tra qua chỉ mục ngược trên toàn
catalogue; nếu hơn một SKU bao phủ được truy vấn thì loại bỏ mẫu đó. 77 truy vấn
mơ hồ đã bị loại theo cách này.

**Khai thác negative khó.** Negative ngẫu nhiên trong lô phần lớn là mặt hàng
không liên quan — phân biệt con bu lông với lon sơn thì encoder học được rất ít.
Negative có giá trị là những SKU mà bộ khớp từ vựng vốn đã nhầm: cùng con bu
lông nhưng dài hơn một cỡ, cùng cái pát nhưng khác mác thép. Script huấn luyện
lấy hai negative loại đó cho mỗi truy vấn từ bộ TF-IDF trước khi train.

## Chạy thử

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py --from-jsonl data/catalog.jsonl
python scripts/build_queries.py
python scripts/evaluate_all.py          # baseline, không cần tải model
python scripts/train.py                 # ~7 phút trên Mac M-series (MPS)
python scripts/significance.py
uvicorn api.main:app --app-dir . --reload
```

Không có model trong `models/`, API vẫn khởi động và phục vụ bằng TF-IDF.

## Giấy phép

MIT — xem [LICENSE](LICENSE).

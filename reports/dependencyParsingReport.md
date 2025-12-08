

# BÁO CÁO THỰC HÀNH LAB 6: PHÂN TÍCH CÚ PHÁP PHỤ THUỘC (DEPENDENCY PARSING)

**Họ và tên:** Nguyễn Công Mạnh

---

## 1. Mục tiêu và Công cụ

Bài thực hành nhằm mục đích làm quen với kỹ thuật Phân tích cú pháp phụ thuộc (Dependency Parsing) để hiểu cấu trúc ngữ pháp của câu thông qua quan hệ Head-Dependent. Công cụ chính được sử dụng là thư viện **spaCy** với mô hình ngôn ngữ tiếng Anh `en_core_web_md` (mô hình tầm trung có hỗ trợ vector và cú pháp đầy đủ).

## 2. Các bước thực hiện và Kết quả

### Phần 1: Cài đặt và Khởi tạo môi trường

* **Thực hiện:** Đã cài đặt thành công thư viện `spacy` và tải mô hình `en_core_web_md` thông qua lệnh pip và python module.
* **Kết quả:** Môi trường đã sẵn sàng, các thư viện phụ thuộc (numpy, requests,...) đều đã thỏa mãn yêu cầu.

### Phần 2: Phân tích câu và Trực quan hóa (Visualization)

* **Thực hiện:**

  * Sử dụng `spacy.load("en_core_web_md")` để tải mô hình.
  * Phân tích câu mẫu: *"The quick brown fox jumps over the lazy dog."*
  * Sử dụng `displacy.serve` để dựng server hiển thị cây cú pháp.
* **Trả lời câu hỏi (Dựa trên kết quả thực nghiệm):**

  1. **Từ gốc (ROOT):** Là từ **"jumps"** (Động từ chính của câu).
  2. **Các từ phụ thuộc của "jumps":** Bao gồm **"fox"** (chủ ngữ - nsubj) và **"over"** (giới từ - prep).
  3. **"fox" là Head của những từ nào:** "fox" là từ điều khiển (Head) của các từ: **"The"** (mạo từ - det), **"quick"** (tính từ - amod), và **"brown"** (tính từ - amod).

### Phần 3: Truy cập thành phần trong cây phụ thuộc

* **Thực hiện:** Viết code duyệt qua từng `token` trong câu *"Apple is looking at buying U.K. startup for $1 billion"*.
* **Kết quả:** Đã trích xuất thành công bảng thông tin gồm: Text, Nhãn phụ thuộc (Dep), Từ cha (Head Text), Loại từ của cha (Head POS) và Danh sách từ con (Children). Ví dụ:

  * *Apple* phụ thuộc vào *looking* với quan hệ `nsubj`.
  * *looking* là `ROOT` của câu.

### Phần 4: Trích xuất thông tin (Information Extraction)

Tôi đã giải quyết hai bài toán trích xuất thông tin dựa trên quan hệ ngữ pháp:

1. **Tìm bộ ba (Chủ ngữ - Động từ - Tân ngữ):**

   * Duyệt qua các token là `VERB`.
   * Tìm trong `token.children` các từ có nhãn `nsubj` (chủ ngữ) và `dobj` (tân ngữ).
   * *Kết quả:* Trích xuất được `(cat, chased, mouse)` và `(dog, watched, them)`.
2. **Tìm tính từ bổ nghĩa cho danh từ:**

   * Duyệt qua các token là `NOUN`.
   * Tìm trong `token.children` các từ có nhãn `amod`.
   * *Kết quả:* Danh từ "cat" có các tính từ bổ nghĩa là `['big', 'fluffy', 'white']`.

### Phần 5: Bài tập tự luyện

Đây là phần nâng cao, tôi đã tự xây dựng các hàm xử lý:

1. **Tìm động từ chính (`find_main_verb`):**

   * Logic: Duyệt token, nếu `token.dep_ == "ROOT"` thì trả về.
   * Kết quả: Tìm được động từ **"decided"** cho câu ví dụ.

2. **Trích xuất cụm danh từ thủ công (`get_noun_chunks_manual`):**

   * Logic: Duyệt token, nếu là `NOUN` hoặc `PROPN` thì lấy `token.subtree`. Có xử lý loại bỏ trường hợp danh từ ghép (`compound`) để tránh trùng lặp.
   * Kết quả: Code thủ công trích xuất được *"The manager"* và *"the meeting"*, tương đồng với kết quả của `spacy.noun_chunks`.

3. **Tìm đường đi ngắn nhất đến ROOT (`get_path_to_root`):**

   * Logic: Sử dụng vòng lặp `while token.head != token` để truy ngược từ từ con lên từ cha cho đến khi gặp ROOT.
   * Kết quả: Tìm được đường đi từ "lazy" đến ROOT: `lazy -> dog -> over -> jumps`.

---

## 3. Các khó khăn và Vấn đề gặp phải

Trong quá trình thực hiện Lab 6, tôi đã gặp và xử lý một số vấn đề sau:

**1. Vấn đề tương thích môi trường với `displacy`:**

* *Mô tả:* Khi chạy lệnh `displacy.serve(doc, style='dep')` trong môi trường Jupyter Notebook/Google Colab, hệ thống báo cảnh báo `UserWarning: [W011] ... calling displacy.serve from within a Jupyter notebook`.
* *Nguyên nhân:* Hàm `serve` cố gắng khởi tạo một web server mới, trong khi môi trường Notebook đã có server riêng.
* *Giải pháp/Bài học:* Theo gợi ý của log lỗi, trong môi trường Notebook nên sử dụng `displacy.render` thay vì `displacy.serve` để hiển thị hình ảnh trực tiếp ngay trong cell code thay vì phải mở tab trình duyệt mới (localhost:5000).

**2. Phức tạp trong logic trích xuất cụm danh từ (Noun Chunks):**

* *Mô tả:* Khi tự viết hàm `get_noun_chunks_manual` ở Bài tập 2, việc chỉ lấy `subtree` của một danh từ là chưa đủ.
* *Khó khăn:* Nếu câu có danh từ ghép (ví dụ: "U.K. startup"), cả "U.K." và "startup" đều là danh từ. Nếu không xử lý kỹ, code sẽ in ra hai cụm chồng chéo nhau.
* *Giải pháp:* Tôi đã phải thêm điều kiện kiểm tra `if token.dep_ == "compound": continue` để bỏ qua các danh từ phụ, chỉ bắt đầu trích xuất từ danh từ chính (Head Noun) của cụm đó.

**3. Xác định quan hệ Head - Dependent:**

* *Khó khăn:* Ban đầu việc xác định đâu là Head, đâu là Child khi nhìn vào câu văn bản thuần túy khá trừu tượng.
* *Giải pháp:* Việc sử dụng thuộc tính `token.head` và `token.children` kết hợp với `displacy` giúp trực quan hóa mối quan hệ này rõ ràng hơn (mũi tên luôn đi từ Head trỏ đến Child).

---

## 4. Kết luận

Buổi thực hành đã giúp tôi nắm vững cách spaCy biểu diễn câu văn dưới dạng cây phụ thuộc. Tôi đã có thể chuyển đổi từ lý thuyết ngữ pháp (Chủ ngữ, Vị ngữ, Bổ ngữ) sang code Python thực tế để trích xuất thông tin tự động. Kỹ năng duyệt cây (Tree Traversal) học được trong bài 3 (tìm đường đi đến Root) là nền tảng quan trọng để xử lý các bài toán NLP phức tạp hơn như tóm tắt văn bản hay trích xuất quan hệ.



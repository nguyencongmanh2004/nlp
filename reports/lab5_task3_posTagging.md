

# BÁO CÁO THỰC HÀNH LAB 5: POS TAGGING VỚI RNN

**Họ và tên:** [Nguyễn Công Mạnh]
**MSSV:** [22001272]
---

## 1. Tổng quan bài toán
Mục tiêu của bài thực hành là xây dựng mô hình mạng nơ-ron hồi quy (RNN) để gán nhãn từ loại (Part-of-Speech Tagging) cho bộ dữ liệu tiếng Anh Universal Dependencies (UD_English-EWT). Bài toán được mô hình hóa dưới dạng *Sequence Labeling* (Gán nhãn chuỗi).

## 2. Quy trình thực hiện (Implementation Details)

### 2.1. Task 1: Tải và Tiền xử lý dữ liệu (Data Preprocessing)
Quá trình xử lý dữ liệu từ định dạng CoNLL-U được thực hiện qua các bước:
* **Đọc dữ liệu:** Sử dụng thư viện `conllu` để parse file. Trích xuất cặp `(form, upos)` tương ứng với `(từ, nhãn)`.
* **Xây dựng từ điển (Vocabulary):**
    * `word_to_idx`: Ánh xạ từ sang index. Thêm token đặc biệt `<PAD>` (index 0) để đệm câu và `<UNK>` (index 1) để xử lý từ lạ (out-of-vocabulary).
    * `tag_to_idx`: Ánh xạ nhãn POS sang index.
    * **Kết quả:** Kích thước từ điển từ là **19,675**, số lượng nhãn POS là **18**.

### 2.2. Task 2: Dataset và DataLoader
Để xử lý việc các câu có độ dài không đồng đều (variable-length sequences), tôi đã thiết kế:
* **Class `POSDataset`:** Kế thừa từ `torch.utils.data.Dataset`, trả về index của từ, index của nhãn và độ dài thực tế của câu.
* **Hàm `collate_fn`:** Đây là bước quan trọng nhất. Sử dụng `torch.nn.utils.rnn.pad_sequence` với `batch_first=True` và `padding_value=0` để đưa các câu trong cùng một batch về cùng độ dài (bằng độ dài câu dài nhất trong batch).

### 2.3. Task 3: Xây dựng mô hình RNN
Mô hình `RNN` được xây dựng với kiến trúc sau:
1.  **Embedding Layer:** Chuyển đổi index của từ thành vector đặc trưng (`embedding_dim=128`).
2.  **RNN Layer:**
    * Sử dụng cơ chế `pack_padded_sequence` trước khi đưa vào RNN. **Lý do:** Giúp mô hình bỏ qua các token `<PAD>`, tăng hiệu suất tính toán và tránh nhiễu từ việc học padding.
    * Sau khi qua RNN, sử dụng `pad_packed_sequence` để khôi phục lại dạng tensor ban đầu.
    * `hidden_size=128`.
3.  **Fully Connected Layer (Linear):** Ánh xạ từ `hidden_size` sang không gian nhãn (`output_size=18`).

### 2.4. Task 4: Huấn luyện (Training)
* **Loss Function:** Sử dụng `nn.CrossEntropyLoss` với tham số `ignore_index=0`. Điều này đảm bảo mô hình không bị phạt lỗi khi dự đoán sai tại các vị trí padding.
* **Optimizer:** `Adam`.
* **Cấu hình:** 30 Epochs, Batch size 64.

---

## 3. Kết quả thực nghiệm và Phân tích

### 3.1. Kết quả định lượng
Mô hình được huấn luyện trong 30 epochs. Dưới đây là bảng tổng hợp kết quả cuối cùng:

| Tập dữ liệu | Loss | Accuracy (Độ chính xác) |
| :--- | :--- | :--- |
| **Train (Epoch 30)** | 0.0149 | 99.52% |
| **Validation (Dev)** | 1.1780 | 86.67% |
| **Test** | **1.1952** | **86.67%** |

### 3.2. Biểu đồ và Phân tích (Analysis)
* **Hiệu năng:** Độ chính xác trên tập Test đạt **86.67%**. Đây là kết quả khá tốt đối với kiến trúc RNN cơ bản (Vanilla RNN) trên bài toán POS Tagging.
* **Hiện tượng Overfitting:**
    * Quan sát log huấn luyện, từ Epoch 10 trở đi, `Train Loss` giảm rất sâu (về gần 0) và `Train Acc` đạt gần tuyệt đối (~99.5%).
    * Tuy nhiên, `Val Loss` bắt đầu tăng dần (từ ~0.6 lên ~1.17), trong khi `Val Acc` đi ngang (bão hòa) ở mức ~86-87%.
    * **Nhận xét:** Mô hình có dấu hiệu Overfitting (học tủ) trên tập train. Mô hình trở nên "tự tin thái quá" (over-confident) vào các dự đoán sai trên tập validation, dẫn đến Loss tăng nhưng Accuracy không đổi.

### 3.3. Demo dự đoán thực tế
Thử nghiệm với câu input: **"i love nlp"**
* **Kết quả mô hình:**
    * `i` $\rightarrow$ `PRON` (Đại từ)
    * `love` $\rightarrow$ `VERB` (Động từ)
    * `nlp` $\rightarrow$ `NOUN` (Danh từ)
* **Đánh giá:** Mô hình dự đoán **chính xác 100%** cấu trúc ngữ pháp của câu ví dụ.

---

## 4. Khó khăn và Giải pháp (Challenges & Solutions)

Trong quá trình thực hiện, tôi đã gặp và giải quyết các vấn đề sau:

1.  **Vấn đề độ dài câu không đồng nhất (Variable Lengths):**
    * *Khó khăn:* Các câu có độ dài khác nhau không thể xếp chồng (stack) thành một Tensor matrix chuẩn.
    * *Giải pháp:* Xây dựng hàm `collate_fn` tùy chỉnh sử dụng `pad_sequence` để thêm padding 0 vào cuối câu ngắn.

2.  **Vấn đề tính toán Loss trên Padding:**
    * *Khó khăn:* Nếu tính Loss cả trên các token padding (số 0), mô hình sẽ bị thiên lệch (bias) về việc dự đoán nhãn 0 vì nó xuất hiện rất nhiều.
    * *Giải pháp:* Sử dụng thuộc tính `ignore_index=0` trong hàm `CrossEntropyLoss` và dùng mask `(y != 0)` khi tính accuracy để loại bỏ hoàn toàn padding khỏi quá trình đánh giá.

3.  **Vấn đề hiệu năng RNN với Padding:**
    * *Giải pháp tối ưu:* Sử dụng kỹ thuật `pack_padded_sequence`. Điều này giúp mạng RNN hiểu được độ dài thực sự của câu (thông qua biến `lengths`), giúp việc lan truyền ngược (backpropagation) chính xác hơn so với việc chỉ feed toàn bộ chuỗi đã padding vào mạng.

## 5. Hướng dẫn chạy code
1.  Đảm bảo cài đặt các thư viện: `torch`, `conllu`, `tensorboard`.
2.  Chỉnh sửa đường dẫn file dataset trong biến `data_file` (nếu cần thiết).
3.  Chạy lần lượt các cell trong Notebook từ trên xuống dưới.
4.  Kết quả huấn luyện sẽ hiển thị trực tiếp và log vào Tensorboard thư mục `run/posTagging`.

---
**Kết luận:** Bài lab đã hoàn thành trọn vẹn các yêu cầu. Mô hình RNN hoạt động ổn định, quy trình xử lý dữ liệu chuẩn xác (xử lý tốt padding/masking) và đạt độ chính xác ~87% trên tập kiểm thử.

---



# **BÁO CÁO LAB 5 – NAMED ENTITY RECOGNITION (NER)**

**Họ và tên:** Nguyễn Công Mạnh
**Mã sinh viên:** 22001272
**Bài thực hành:** Lab 5 - Named Entity Recognition (NER)

---

## 1. Giới thiệu và Mục tiêu (Introduction)

Bài thực hành nhằm mục tiêu xây dựng hệ thống Nhận dạng thực thể tên (Named Entity Recognition - NER) sử dụng kiến trúc mạng nơ-ron hồi quy (RNN). Mô hình có nhiệm vụ phân loại từng token trong câu vào các nhãn thực thể chuẩn IOB (Inside, Outside, Beginning) như `PER` (Person), `LOC` (Location), `ORG` (Organization), và `MISC`.

Dữ liệu được sử dụng là **CoNLL-2003**, một bộ dữ liệu benchmark tiêu chuẩn cho bài toán NER.

---

## 2. Chuẩn bị và Tiền xử lý dữ liệu (Data Preparation)

### 2.1. Tải và Khám phá dữ liệu

* **Nguồn dữ liệu:** Sử dụng thư viện `datasets` của Hugging Face để tải `conll2003`.
* **Cấu trúc dữ liệu:** Gồm các tập `train` (14,041 dòng), `validation` (3,250 dòng) và `test` (3,453 dòng).
* **Hệ thống nhãn:** Dữ liệu gốc sử dụng nhãn số. Chúng tôi đã ánh xạ sang dạng chuỗi gồm 9 nhãn:
  `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`.

### 2.2. Xây dựng Vocabulary

* **Word Dictionary:** Xây dựng từ điển ánh xạ từ → index. Thêm token đặc biệt `<PAD>` (index 0) và `<UNK>` (index 1).
* **Kích thước từ điển (Vocab Size):** **23,625** từ duy nhất được trích xuất từ tập train.
* **NER Tag Dictionary:** Ánh xạ 9 nhãn thực thể sang index số nguyên để tính toán Loss.

---

## 3. Kiến trúc Mô hình và Dataloader (Implementation)

### 3.1. Dataloader và Kỹ thuật Padding

* **Custom Dataset:** Lớp `NERDataset` được xây dựng để trả về cặp tensor `(words, ner_tags)`.
* **Collate Function:** Sử dụng `collate_fn` kết hợp `pad_sequence` để đưa các câu về cùng độ dài.
  Giá trị padding = `0`.
* **Batch Size:** 64.

### 3.2. Mô hình RNN (Task 2 & 3)

Mô hình `SimpleRNN` gồm các thành phần:

1. **Embedding Layer**

   * Input dimension: 23,625
   * Embedding dimension: 256

2. **RNN Layer**

   * Hidden dimension: 256
   * Sử dụng `pack_padded_sequence` và `pad_packed_sequence` để bỏ qua padding khi tính toán.

3. **Fully Connected Layer**

   * Ánh xạ từ hidden state → 9 classes.

(*Lưu ý:* Đây là mô hình RNN một chiều – phần phân tích sẽ đề cập lý do nên nâng cấp lên Bi-LSTM.)

---

## 4. Quá trình Huấn luyện (Training Process)

### 4.1. Cấu hình huấn luyện

* **Optimizer:** Adam
* **Loss Function:** `CrossEntropyLoss(ignore_index=0)`
* **Device:** GPU (CUDA) hoặc CPU
* **Số epochs:** Huấn luyện nhiều vòng cho đến khi hội tụ.

### 4.2. Diễn biến Loss và Accuracy

* **Khởi đầu:** Loss ~0.7193, Accuracy ~82.5%
* **Kết thúc:** Loss giảm xuống **0.0048**, Accuracy trên tập train đạt **99.81%**

*Mô hình hội tụ tốt, loss giảm đều.*

---

## 5. Đánh giá và Phân tích Kết quả (Evaluation & Analysis)

### 5.1. Kết quả định lượng trên tập Validation

* **Validation Loss:** 0.5635
* **Validation Accuracy:** 91.71%
* **Precision:** 53.80%
* **Recall:** 74.74%
* **F1-score:** 62.56%

### 5.2. Phân tích chi tiết theo từng loại thực thể

| Entity   | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| **LOC**  | 0.67      | 0.83   | 0.74     | 1837    |
| **PER**  | 0.83      | 0.68   | 0.74     | 1842    |
| **MISC** | 0.53      | 0.70   | 0.60     | 922     |
| **ORG**  | 0.32      | 0.77   | 0.45     | 1341    |

#### Nhận xét:

1. **Overfitting rõ rệt:**
   Train accuracy = 99.8% vs Val F1 = 62.5%.

2. **Recall cao hơn Precision:**
   Mô hình bắt được nhiều thực thể nhưng dự đoán sai nhãn khá nhiều.

3. **Nhãn ORG yếu nhất:**
   F1 = 0.45, vì tổ chức thường có tên phức tạp và dễ nhầm với LOC/PER nếu mô hình không có ngữ cảnh hai chiều.

### 5.3. Thử nghiệm thực tế

* “John lives in New York.” → `['B-PER', '0', '0', 'B-LOC', 'I-LOC']`
* “VNU University is located in Hanoi” → `['B-ORG', 'I-ORG', '0', '0', '0', 'B-LOC']`

Cả hai đều dự đoán đúng.

---

## 6. Khó khăn và Giải pháp (Challenges & Solutions)

1. **Độ dài câu không đồng nhất**
   → Dùng `pad_sequence` và `pack_padded_sequence`.

2. **Mất cân bằng nhãn**
   → Dùng `ignore_index=0` và đánh giá bằng F1-score.

3. **Giới hạn của Simple RNN**
   → Đề xuất nâng cấp lên **Bi-LSTM + CRF** để:

   * hiểu được ngữ cảnh hai chiều,
   * tăng F1 của ORG,
   * giảm overfitting.

---

## 7. Kết luận

Bài thực hành đã hoàn thiện pipeline hoàn chỉnh cho bài toán NER.
Mặc dù mô hình SimpleRNN đạt F1 = **62.56%**, nhưng kết quả cho thấy cần sử dụng các mô hình mạnh hơn như **Bi-LSTM** hoặc **Bi-LSTM + CRF** để cải thiện hiệu suất.

### **Kết quả cuối cùng trên tập Validation:**

* **Accuracy:** 91.71%
* **F1-score:** 62.56%

---

## **Hướng dẫn chạy code**

1. Cài đặt thư viện:
   `pip install datasets seqeval torch`
2. Chạy Jupyter Notebook đính kèm.
3. Load model đã train và dùng hàm `predict_sentence` để dự đoán câu mới.



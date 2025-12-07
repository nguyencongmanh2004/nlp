
**Họ và tên:** Nguyễn Công Mạnh
**Mã sinh viên:** 22001272
**Lab:** 5 - Named Entity Recognition (NER)

---

## 1. Giới thiệu và Mục tiêu (Introduction)

Mục tiêu của bài thực hành là xây dựng hệ thống Nhận dạng thực thể tên (Named Entity Recognition - NER) sử dụng mạng nơ-ron hồi quy. Bài toán yêu cầu mô hình gán nhãn cho từng từ trong câu theo định dạng IOB (Inside, Outside, Beginning) với các thực thể: Người (PER), Tổ chức (ORG), Địa điểm (LOC), và Khác (MISC).

Trong báo cáo này, tôi trình bày việc cài đặt và so sánh hai kiến trúc: **Simple RNN** và **Bidirectional LSTM (Bi-LSTM)** trên bộ dữ liệu chuẩn **CoNLL-2003**.

---

## 2. Chuẩn bị dữ liệu (Data Preparation) [Đáp ứng Task 1]

### 2.1. Bộ dữ liệu CoNLL-2003

* **Nguồn dữ liệu:** Sử dụng thư viện `datasets` của Hugging Face để tải `conll2003`.
* **Cấu trúc:** Dữ liệu bao gồm các câu đã được tách từ (tokenized) và gán nhãn NER.
* **Phân chia:** Tập dữ liệu được chia sẵn thành 3 phần: Train (14,041 dòng), Validation (3,250 dòng), và Test (3,453 dòng).

### 2.2. Tiền xử lý (Preprocessing)

Để mô hình có thể học được, tôi đã thực hiện các bước tiền xử lý sau:

1. **Xây dựng bộ từ điển (Vocabulary):**

   * Duyệt qua toàn bộ tập Train để tạo từ điển ánh xạ từ `word` sang `index`.
   * Thêm các token đặc biệt: `<PAD>` (index 0) để đệm câu, và `<UNK>` (index 1) để xử lý các từ lạ không xuất hiện trong tập train.
   * **Kích thước bộ từ điển:** 23,625 từ duy nhất.

2. **Xử lý nhãn (Label Encoding):**

   * Chuyển đổi nhãn từ dạng số (0-8) sang dạng chuỗi (B-PER, I-PER, ...) và ngược lại để tiện cho việc đánh giá.

3. **Custom Dataset & DataLoader:**

   * Xây dựng lớp `NERDataset` kế thừa từ `torch.utils.data.Dataset`.
   * Cài đặt hàm `collate_fn` sử dụng `pad_sequence` để đưa các câu trong cùng một batch về cùng độ dài. Giá trị padding cho token là 0 và cho nhãn NER cũng là 0 (để `CrossEntropyLoss` bỏ qua sau này).

---

## 3. Xây dựng Mô hình (Model Architecture) [Đáp ứng Task 2]

Tôi đã cài đặt hai mô hình để so sánh hiệu quả, trong đó tập trung vào **Bi-LSTM** theo yêu cầu nâng cao của đề bài.

### 3.1. Simple RNN (Baseline)

* Sử dụng lớp `nn.RNN` cơ bản.
* Cấu trúc: Embedding Layer → Simple RNN → Fully Connected Layer.

### 3.2. Bi-LSTM (Mô hình chính)

Mô hình Bi-LSTM được lựa chọn vì khả năng nắm bắt ngữ cảnh từ cả hai phía (quá khứ và tương lai) của từ hiện tại, điều này rất quan trọng trong bài toán NER.

* **Kiến trúc chi tiết:**

  * **Input:** Batch các câu đã được padding.
  * **Embedding Layer:** Chuyển đổi index của từ thành vector (Kích thước vocab: 23,625, Embedding dim: 256).
  * **Bi-LSTM Layer:** Sử dụng `nn.LSTM` với `bidirectional=True`.

    * Hidden dimension: 256.
    * Vì là 2 chiều, đầu ra tại mỗi bước thời gian sẽ là sự kết hợp của chiều thuận và chiều nghịch (kích thước (2 \times hidden_dim)).
  * **Handling Variable Lengths:** Sử dụng `pack_padded_sequence` trước khi đưa vào LSTM và `pad_packed_sequence` sau đầu ra LSTM. Kỹ thuật này giúp mô hình bỏ qua tính toán trên các token padding, tăng tốc độ và độ chính xác.
  * **Fully Connected Layer:** Ánh xạ từ không gian ẩn (512 chiều) về số lượng nhãn NER để phân loại.

---

## 4. Quá trình Huấn luyện (Training) [Đáp ứng Task 3]

* **Siêu tham số (Hyperparameters):**

  * Optimizer: `Adam`
  * Loss Function: `CrossEntropyLoss` với `ignore_index=0`
  * Batch size: 64
  * Epochs: 15 (Bi-LSTM), 50 (RNN)
* **Môi trường:** Training trên GPU (CUDA)

**Diễn biến Loss:**

* Loss giảm đều theo thời gian.
* Với Bi-LSTM, loss giảm xuống ~0.0004 và train accuracy đạt 100% ở các epoch cuối.

---

## 5. Đánh giá và Phân tích Kết quả (Evaluation & Analysis) [Đáp ứng Task 3 & 4]

Sử dụng thư viện `seqeval` để tính Precision, Recall và F1-score theo chuẩn CoNLL.

### 5.1. Kết quả trên tập Test

| Metric        | Simple RNN | **Bi-LSTM (Main Model)** |
| ------------- | ---------- | ------------------------ |
| **Accuracy**  | 94.19%     | **95.09%**               |
| **Precision** | 0.656      | **0.722**                |
| **Recall**    | 0.732      | **0.768**                |
| **F1-Score**  | 0.692      | **0.744**                |

### 5.2. Phân tích chi tiết (Bi-LSTM)

| Entity   | Precision | Recall | F1-score | Nhận xét                               |
| -------- | --------- | ------ | -------- | -------------------------------------- |
| **LOC**  | 0.67      | 0.87   | 0.76     | Recall cao nhưng precision trung bình. |
| **MISC** | 0.82      | 0.70   | 0.76     | Độ chính xác tốt.                      |
| **ORG**  | 0.65      | 0.70   | 0.67     | Lớp khó nhất.                          |
| **PER**  | 0.81      | 0.75   | 0.78     | Nhận diện tốt nhất.                    |

---

## 6. Demo Dự đoán (Inference)

1. **"John lives in New York."**

   * Dự đoán: `['B-PER', 'O', 'O', 'B-LOC', 'I-LOC']`

2. **"The FIFA World Cup will be held in Canada."**

   * Dự đoán: `['O', 'B-MISC', 'B-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'B-LOC']`

3. **"VNU University is located in Hanoi"**

   * Dự đoán: `['B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-LOC']`

---

## 7. Khó khăn và Giải pháp (Challenges & Solutions)

1. **Độ dài câu khác nhau → dùng `pad_sequence` + `pack_padded_sequence`.**
2. **Padding gây nhiễu Loss → dùng `ignore_index=0`.**
3. **Lỗi khi load CoNLL → dùng `trust_remote_code=True`.**

---

## 8. Kết luận

Bài thực hành hoàn thành đầy đủ yêu cầu. Mô hình Bi-LSTM đạt **F1-score = 0.744** trên tập Test, cao hơn đáng kể so với Simple RNN. Trong tương lai có thể cải thiện bằng Bi-LSTM-CRF.

**Kết quả xác nhận cuối:**

* Accuracy (Test): **0.9509**
* Dự đoán câu: **VNU University is located in Hanoi → [B-ORG, I-ORG, O, O, O, B-LOC]**


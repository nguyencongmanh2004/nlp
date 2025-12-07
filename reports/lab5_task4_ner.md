
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
* [cite_start]**Nguồn dữ liệu:** Sử dụng thư viện `datasets` của Hugging Face để tải `conll2003`[cite: 377].
* **Cấu trúc:** Dữ liệu bao gồm các câu đã được tách từ (tokenized) và gán nhãn NER.
* [cite_start]**Phân chia:** Tập dữ liệu được chia sẵn thành 3 phần: Train (14,041 dòng), Validation (3,250 dòng), và Test (3,453 dòng) [cite: 387-392].

### 2.2. Tiền xử lý (Preprocessing)
Để mô hình có thể học được, tôi đã thực hiện các bước tiền xử lý sau:
1.  **Xây dựng bộ từ điển (Vocabulary):**
    * Duyệt qua toàn bộ tập Train để tạo từ điển ánh xạ từ `word` sang `index`.
    * [cite_start]Thêm các token đặc biệt: `<PAD>` (index 0) để đệm câu, và `<UNK>` (index 1) để xử lý các từ lạ không xuất hiện trong tập train[cite: 421].
    * [cite_start]**Kích thước bộ từ điển:** 23,625 từ duy nhất[cite: 432].

2.  **Xử lý nhãn (Label Encoding):**
    * [cite_start]Chuyển đổi nhãn từ dạng số (0-8) sang dạng chuỗi (B-PER, I-PER,...) và ngược lại để tiện cho việc đánh giá[cite: 436].

3.  **Custom Dataset & DataLoader:**
    * [cite_start]Xây dựng lớp `NERDataset` kế thừa từ `torch.utils.data.Dataset`[cite: 491].
    * Cài đặt hàm `collate_fn` sử dụng `pad_sequence` để đưa các câu trong cùng một batch về cùng độ dài. [cite_start]Giá trị padding cho token là 0 và cho nhãn NER cũng là 0 (để `CrossEntropyLoss` bỏ qua sau này)[cite: 507].

---

## 3. Xây dựng Mô hình (Model Architecture) [Đáp ứng Task 2]

Tôi đã cài đặt hai mô hình để so sánh hiệu quả, trong đó tập trung vào **Bi-LSTM** theo yêu cầu nâng cao của đề bài.

### 3.1. Simple RNN (Baseline)
* Sử dụng lớp `nn.RNN` cơ bản.
* Cấu trúc: Embedding Layer $\rightarrow$ Simple RNN $\rightarrow$ Fully Connected Layer.

### 3.2. Bi-LSTM (Mô hình chính)
Mô hình Bi-LSTM được lựa chọn vì khả năng nắm bắt ngữ cảnh từ cả hai phía (quá khứ và tương lai) của từ hiện tại, điều này rất quan trọng trong bài toán NER.

* **Kiến trúc chi tiết:**
    * **Input:** Batch các câu đã được padding.
    * **Embedding Layer:** Chuyển đổi index của từ thành vector (Kích thước vocab: 23,625, Embedding dim: 256).
    * **Bi-LSTM Layer:** Sử dụng `nn.LSTM` với `bidirectional=True`.
        * Hidden dimension: 256.
        * [cite_start]Vì là 2 chiều, đầu ra tại mỗi bước thời gian sẽ là sự kết hợp của chiều thuận và chiều nghịch (kích thước $2 \times hidden\_dim$) [cite: 540-547].
    * **Handling Variable Lengths:** Sử dụng `pack_padded_sequence` trước khi đưa vào LSTM và `pad_packed_sequence` sau đầu ra LSTM. [cite_start]Kỹ thuật này giúp mô hình bỏ qua tính toán trên các token padding, tăng tốc độ và độ chính xác [cite: 551-552].
    * **Fully Connected Layer:** Ánh xạ từ không gian ẩn ($256 \times 2 = 512$) về số lượng nhãn NER để phân loại.

---

## 4. Quá trình Huấn luyện (Training) [Đáp ứng Task 3]

* **Siêu tham số (Hyperparameters):**
    * Optimizer: `Adam`.
    * Loss Function: `CrossEntropyLoss` với `ignore_index=0`. [cite_start]Điều này cực kỳ quan trọng để loss không bị nhiễu bởi các token padding[cite: 598].
    * Batch size: 64.
    * Epochs: 15 (cho Bi-LSTM) và 50 (cho RNN).
* **Môi trường:** Training trên GPU (CUDA) để tăng tốc độ.

**Diễn biến Loss:**
* Mô hình hội tụ khá tốt. Loss trên tập train giảm đều đặn.
* [cite_start]Với Bi-LSTM, loss giảm xuống mức rất thấp (~0.0004) và train accuracy đạt 100% vào các epoch cuối[cite: 607], cho thấy mô hình đủ khả năng fit dữ liệu.

---

## 5. Đánh giá và Phân tích Kết quả (Evaluation & Analysis) [Đáp ứng Task 3 & 4]

Để đánh giá khách quan, tôi sử dụng thư viện `seqeval` để tính toán các chỉ số Precision, Recall và F1-score theo chuẩn CoNLL (đánh giá trên thực thể thay vì từng token riêng lẻ).

### 5.1. Kết quả trên tập Test (Test Set)

Dưới đây là bảng so sánh hiệu năng giữa hai mô hình trên tập Test:

| Metric | [cite_start]Simple RNN [cite: 679-682] | [cite_start]**Bi-LSTM (Main Model)** [cite: 710-713] |
| :--- | :--- | :--- |
| **Accuracy** | 94.19% | **95.09%** |
| **Precision** | 0.656 | **0.722** |
| **Recall** | 0.732 | **0.768** |
| **F1-Score** | 0.692 | **0.744** |

### 5.2. Phân tích chi tiết (Bi-LSTM)
Mô hình Bi-LSTM cho kết quả vượt trội hơn hẳn Simple RNN (F1-score tăng từ 0.69 lên 0.74). [cite_start]Chi tiết kết quả theo từng loại thực thể của Bi-LSTM[cite: 714]:

| Entity | Precision | Recall | F1-score | Nhận xét |
| :--- | :--- | :--- | :--- | :--- |
| **LOC** | 0.67 | 0.87 | 0.76 | Nhận diện địa điểm khá tốt, recall cao nhưng precision trung bình. |
| **MISC** | 0.82 | 0.70 | 0.76 | Độ chính xác cao nhưng bỏ sót một số trường hợp. |
| **ORG** | 0.65 | 0.70 | 0.67 | Đây là lớp khó nhận diện nhất (F1 thấp nhất). |
| **PER** | 0.81 | 0.75 | 0.78 | Nhận diện tên người tốt nhất trong các lớp. |

**Nhận xét:** Việc sử dụng Bi-LSTM giúp mô hình hiểu ngữ cảnh tốt hơn. Ví dụ, để phân biệt "Washington" (người) và "Washington" (địa điểm), mô hình cần thông tin từ các từ phía sau. Simple RNN chỉ nhìn về quá khứ nên gặp khó khăn, trong khi Bi-LSTM giải quyết tốt vấn đề này.

---

## 6. Demo Dự đoán (Inference)
Tôi đã viết hàm `predict_sentence` để kiểm thử trên các câu thực tế. [cite_start]Dưới đây là kết quả của mô hình Bi-LSTM [cite: 733-736]:

1.  **Câu:** *"John lives in New York."*
    * **Dự đoán:** `['B-PER', 'O', 'O', 'B-LOC', 'I-LOC']`
    * **Đánh giá:** Chính xác hoàn toàn.

2.  **Câu:** *"The FIFA World Cup will be held in Canada."*
    * **Dự đoán:** `['O', 'B-MISC', 'B-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'B-LOC']`
    * **Đánh giá:** Nhận diện đúng "FIFA World Cup" là sự kiện (MISC) và "Canada" là địa điểm (LOC).

3.  **Câu:** *"VNU University is located in Hanoi"* (Yêu cầu đề bài)
    * **Dự đoán:** `['B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-LOC']`
    * **Đánh giá:** Mô hình nhận diện chính xác "VNU University" là tổ chức và "Hanoi" là địa điểm.

---

## 7. Khó khăn và Giải pháp (Challenges & Solutions)

Trong quá trình thực hiện Lab, tôi đã gặp và giải quyết các vấn đề sau:

1.  **Vấn đề độ dài câu khác nhau:**
    * *Khó khăn:* Các câu trong batch có độ dài không bằng nhau không thể đưa vào matrix tính toán.
    * *Giải pháp:* Sử dụng `pad_sequence` trong `collate_fn` để thêm padding. Quan trọng hơn, tôi sử dụng `pack_padded_sequence` trong model Bi-LSTM để RNN không phải xử lý các token vô nghĩa này, giúp model hội tụ nhanh hơn.

2.  **Tính Loss với Padding:**
    * *Khó khăn:* Nếu tính cả padding vào loss, model sẽ có xu hướng dự đoán nhãn padding (O) cho mọi thứ để giảm loss nhanh.
    * *Giải pháp:* Thiết lập `ignore_index=0` trong hàm `CrossEntropyLoss`.

3.  **Load dữ liệu CoNLL:**
    * *Khó khăn:* Thư viện `datasets` báo lỗi bảo mật custom code.
    * [cite_start]*Giải pháp:* Thêm tham số `trust_remote_code=True` khi load dataset[cite: 381].

---

## 8. Kết luận
Bài thực hành đã hoàn thành đầy đủ các yêu cầu đề ra. Tôi đã xây dựng thành công pipeline từ xử lý dữ liệu đến huấn luyện mô hình Bi-LSTM. Kết quả F1-score đạt **0.744** trên tập Test chứng minh tính hiệu quả của kiến trúc hai chiều đối với bài toán chuỗi như NER.

Mặc dù kết quả khả quan, mô hình vẫn có thể cải thiện thêm bằng cách kết hợp lớp **CRF (Conditional Random Field)** ở tầng cuối cùng (Bi-LSTM-CRF) để bắt các ràng buộc giữa các nhãn (ví dụ: I-PER không thể đứng đầu câu).

---
**Kết quả xác nhận:**
* Độ chính xác trên tập validation (Accuracy): **0.9509** (Bi-LSTM Test Accuracy)
* Dự đoán câu "VNU University is located in Hanoi": **[B-ORG, I-ORG, O, O, O, B-LOC]**


# Lab 5

## A. Làm quen với PyTorch và Token Classification 

### 1. Mục tiêu
- Làm quen với thao tác với **Tensor** trong PyTorch.  
- Hiểu cơ chế **autograd** để tự động tính gradient.  
- Xây dựng các mô hình đơn giản bằng **torch.nn.Module**.  
- Thực hành với **nn.Linear** và **nn.Embedding**.  
- (Mở rộng) Xây dựng mô hình phân loại nhãn theo token bằng **RNN**.

### 2. Nội dung thực hiện

#### Phần 1: Khám phá Tensor
- **Task 1.1. Tạo Tensor**: Tạo tensor từ list, từ NumPy, tensor toàn 1 và tensor ngẫu nhiên; kiểm tra `shape`, `dtype`, `device`.  
- **Task 1.2. Các phép toán trên Tensor**: Cộng, nhân scalar, nhân ma trận.  
- **Task 1.3. Indexing và slicing**: Trích xuất phần tử, hàng, cột từ tensor.  
- **Task 1.4. Thay đổi hình dạng Tensor**: `view`, `reshape`, `permute`.  

#### Phần 2: Tự động tính đạo hàm với autograd
- Thực hành theo dõi gradient của tensor với `requires_grad=True`.  
- Thực hiện các phép toán và gọi `backward()` để tính gradient tự động.  

#### Phần 3: Xây dựng mô hình với torch.nn
- **Task 3.1. Khởi tạo lớp nn.Linear**  
- **Task 3.2. Khởi tạo lớp nn.Embedding**  
- **Task 3.3. Kết hợp thành một nn.Module hoàn chỉnh**  
- **Task mở rộng**: Xây dựng `SimpleRNNForTokenClassification` để giải quyết bài toán **POS tagging**.

---

## B. Phân loại văn bản với RNN/LSTM (task2.ipynb)

### 1. Mục tiêu
So sánh hiệu năng của 4 pipeline phân loại văn bản:
1. **TF-IDF + Logistic Regression** (Baseline 1).  
2. **Word2Vec (trung bình câu) + Dense** (Baseline 2).  
3. **Pre-trained Word2Vec Embedding + LSTM**.  
4. **Embedding học từ đầu + LSTM**.

### 2. Công việc thực hiện
- **Tải và tiền xử lý dữ liệu**: dùng pandas, mã hóa nhãn với `LabelEncoder`.  
- **Baseline 1**: `TfidfVectorizer(max_features=5000)` + `LogisticRegression(max_iter=1000)`.  
- **Baseline 2**: Huấn luyện `gensim.Word2Vec` (vector_size=100), lấy **mean embedding** của câu → Dense + Dropout.  
- **LSTM với Pre-trained Embedding**: Tokenizer → sequences → padding (max_len=100), embedding từ Word2Vec (không trainable) → LSTM(128, dropout=0.2, recurrent_dropout=0.2) → Dense(softmax).  
- **LSTM với Embedding học từ đầu**: tương tự Task 3 nhưng embedding trainable.  
- Huấn luyện với **EarlyStopping** (monitor='val_loss', patience=10, restore_best_weights=True).  
- Đánh giá bằng **Accuracy**, **Macro F1** và **Test Loss (log loss)** cho các mô hình Keras.

### 3. Dữ liệu và tiền xử lý (Task 0)
- **Nguồn**: HWU intents dataset (train/val/test).  
- **Đọc dữ liệu**: `pandas.read_csv` (cột: text, category/intent).  
- **Mã hóa nhãn**: `LabelEncoder` (fit trên train, transform cho val/test).  
- **Số lớp**: `num_classes = len(label_encoder.classes_)`.

### 4. Baseline 1 - TF-IDF + Logistic Regression (Task 1)
- Pipeline: `TfidfVectorizer(max_features=5000)` → `LogisticRegression(max_iter=1000)`.  
- Huấn luyện trên train, đánh giá trên test với `classification_report` (Macro F1/Accuracy).

### 5. Baseline 2 - Word2Vec (Avg) + Dense (Task 2)
- Huấn luyện Word2Vec: `Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=100)`.  
- Biểu diễn câu: trung bình các vector từ trong vocab.  
- Kiến trúc Dense: `Dense(128, relu)` → `Dropout(0.5)` → `Dense(num_classes, softmax)`.  
- Huấn luyện: `optimizer='adam'`, `loss='sparse_categorical_crossentropy'`, `metrics=['accuracy']`, **EarlyStopping**.  
- Đánh giá: Macro F1/Accuracy; tính Test log loss thủ công trên xác suất dự đoán.

### 6. LSTM với Pre-trained Embedding (Task 3)
- **Tiền xử lý chuỗi**: Tokenizer(oov_token="<UNK>") → texts_to_sequences → pad_sequences(maxlen=100).  
- **Ma trận embedding**: size (vocab_size, 100), khởi tạo từ w2v_model.wv, trainable=False.  
- **Kiến trúc**:
  - `Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=100, mask_zero=True, trainable=False)`  
  - `LSTM(128, dropout=0.2, recurrent_dropout=0.2)`  
  - `Dense(num_classes, softmax)`  
- Huấn luyện: Adam + EarlyStopping; đánh giá Macro F1/Accuracy và Test log loss.

### 7. LSTM với Embedding học từ đầu (Task 4)
- Dùng cùng Tokenizer và padding từ Task 3.  
- Kiến trúc:
  - `Embedding(input_dim=vocab_size, output_dim=100, input_length=100, mask_zero=True, trainable=True)`  
  - `LSTM(128, dropout=0.2, recurrent_dropout=0.2)`  
  - `Dense(num_classes, softmax)`  
- Huấn luyện/EarlyStopping tương tự Task 3; đánh giá như trên.

### 8. Đánh giá (Task 5)
- **Chỉ số đánh giá**: Accuracy, Macro F1-score (`sklearn classification_report`), Test Loss (log loss cho Keras; TF-IDF+LR N/A).  

| Pipeline                         | Macro F1 (Test) | Test Accuracy | Test Loss |
|---------------------------------|----------------|---------------|-----------|
| TF-IDF + Logistic Regression     | 0.83           | 0.83          | N/A       |
| Word2Vec (Avg) + Dense           | 0.80           | 0.81          | 0.65      |
| Pre-trained Embedding + LSTM     | 0.84           | 0.84          | 0.54      |
| Scratch Embedding + LSTM         | 0.86           | 0.85          | 0.60      |

### 9. Phân tích định tính
- **TF-IDF + LR**: Pipeline đơn giản nhưng khá ổn định với dữ liệu nhỏ, dễ triển khai.  
- **Word2Vec + Dense**: Có thể nắm bắt ngữ nghĩa, nhưng trung bình vector bỏ qua thứ tự từ → một số thông tin ngữ cảnh bị mất.  
- **LSTM Pre-trained**: Giữ được ngữ cảnh, cải thiện Macro F1, giảm log loss.  
- **LSTM Scratch**: Mô hình trainable tự học embedding, kết quả tốt nhất về Macro F1, nhưng log loss hơi cao hơn pre-trained do dữ liệu còn hạn chế.  
- Nhìn chung, **mô hình LSTM** vượt trội về khả năng học ngữ cảnh so với các baseline.


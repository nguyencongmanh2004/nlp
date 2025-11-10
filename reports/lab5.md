# Lab 5: Text Classification - Report and Analysis

## 1. Quá trình thực hiện

### Task 1: Implement `TextClassifier`
- Triển khai lớp `TextClassifier` sử dụng Logistic Regression từ scikit-learn.
- Chuẩn hóa dữ liệu đầu vào bằng `CountVectorizer` kết hợp với `RegexTokenizer`.
- Thực hiện các phương thức:
  - `fit`: huấn luyện mô hình trên tập huấn luyện.
  - `predict`: dự đoán nhãn trên tập kiểm tra.
  - `evaluate`: tính các chỉ số Accuracy, Precision, Recall và F1-score.
- Kết quả thử nghiệm ban đầu trên tập dữ liệu nhỏ cho thấy mô hình cơ bản chưa tối ưu (Accuracy = 0.1667, Precision = 0.0, Recall = 0.0, F1 = 0.0), nguyên nhân chủ yếu do tập dữ liệu quá nhỏ và không cân bằng.

### Task 2: Basic Test Case
- Tạo tập test để tách dữ liệu train/test, huấn luyện mô hình và đánh giá kết quả.
- Mô hình Logistic Regression cơ bản được thử trên tập dữ liệu nhỏ gồm 30 văn bản (15 positive, 15 negative) với tỷ lệ train/test = 0.8/0.2.
- Kết quả đánh giá bước đầu phản ánh hiệu quả hạn chế, đặc biệt Precision và Recall bằng 0, cho thấy mô hình chưa học được đặc trưng tốt từ dữ liệu.

### Task 3: Spark ML Pipeline
- Chạy script `lab5_spark_sentiment_analysis.py` sử dụng Spark ML.
- Các bước xử lý:
  1. Tokenization và lọc stopwords.
  2. Tạo vector đặc trưng bằng Word2Vec và TF-IDF.
  3. Chia dữ liệu train/test.
  4. Huấn luyện các mô hình: Naive Bayes, Logistic Regression (dense/sparse), và MLP.
- **Kết quả đạt được:**
  - Naive Bayes Accuracy: 0.6871
  - Logistic Regression (dense): 0.7196
  - Logistic Regression (sparse): 0.7286
  - MLP scikit-learn Accuracy: 0.6911
- Nhận xét: Logistic Regression sparse và dense đều cải thiện rõ rệt so với mô hình baseline, MLP chưa tối ưu hoàn toàn do cần điều chỉnh hyperparameter.

### Task 4: Model Improvement Experiment
- Thử cải thiện mô hình bằng cách:
  - Sử dụng Word2Vec để tạo vector dense thay vì chỉ dùng CountVectorizer.
  - Sử dụng Logistic Regression với TF-IDF features thay vì chỉ Bag-of-Words.
  - Thử MLP với 2 hidden layers.
- Kết quả cho thấy mô hình cải tiến (Logistic Regression sparse với TF-IDF) tăng Accuracy từ 0.1667 lên khoảng 0.7286, F1-score cũng cải thiện đáng kể.

## 2. Phân tích kết quả
- **Mô hình baseline**: Logistic Regression với CountVectorizer hoạt động kém trên tập dữ liệu nhỏ, do thiếu thông tin đặc trưng.
- **Mô hình cải tiến**: Logistic Regression với TF-IDF hoặc Word2Vec vector hóa giúp mô hình nắm bắt tốt hơn ngữ cảnh và tần suất từ, tăng Accuracy và F1-score.
- **So sánh mô hình**:
  - Logistic Regression sparse tốt hơn Naive Bayes và MLP trên tập dữ liệu này.
  - MLP có tiềm năng nhưng cần tuning (số layer, số neuron, learning rate, batch size).

## 3. Thách thức và giải pháp
- **Dữ liệu nhỏ và không cân bằng**: Gây ra kết quả Precision/Recall thấp. Giải pháp: mở rộng dữ liệu hoặc sử dụng oversampling/undersampling.
- **Xử lý đặc trưng văn bản**: CountVectorizer đơn giản không thể nắm bắt ngữ nghĩa. Giải pháp: dùng TF-IDF, Word2Vec hoặc embeddings khác.
- **Tối ưu mô hình**: MLP gặp cảnh báo chưa hội tụ. Giải pháp: tăng số epoch, điều chỉnh learning rate hoặc dùng mô hình khác.

## 4. Hướng cải thiện trong tương lai
- Thử các mô hình phức tạp hơn như BERT hoặc Transformer cho ngữ nghĩa sâu hơn.
- Sử dụng grid search hoặc random search để tối ưu hyperparameters.
- Augment dữ liệu để tăng số lượng văn bản huấn luyện.
- Kết hợp nhiều đặc trưng: n-gram, POS tagging, sentiment lexicons.
- Đánh giá trên tập dữ liệu lớn hơn để có kết quả đáng tin cậy.

## 5. Kết luận
- Mô hình Logistic Regression với TF-IDF/Word2Vec cho kết quả tốt hơn nhiều so với baseline.
- Cải tiến đặc trưng và chọn mô hình phù hợp là yếu tố quan trọng để nâng cao Accuracy và F1-score.
- Các phương pháp nâng cao hơn như MLP, embeddings sâu hơn hoặc mô hình Transformer có thể được thử để cải thiện hiệu quả trên dữ liệu thực tế.

## 6. Tài liệu tham khảo
- PySpark MLlib documentation: https://spark.apache.org/docs/latest/ml-guide.html

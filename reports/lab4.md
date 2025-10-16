# BÁO CÁO LAB 4: WORD EMBEDDINGS WITH WORD2VEC

**Sinh viên:** Nguyễn Công Mạnh
**Ngày thực hiện:** 14/10/2025  

---

## 1.1. Task 1: Setup

### **Mục tiêu**
Thiết lập môi trường làm việc cho bài lab và chuẩn bị mô hình embedding đã huấn luyện sẵn để sử dụng trong các task sau.

---

### **Quy trình thực hiện**

1. **Cài đặt thư viện gensim**

2. **Tải mô hình embedding có sẵn**  
   - Mô hình sử dụng: **`glove-wiki-gigaword-50`**.  
   - Sau khi tải lần đầu, mô hình được lưu trữ trong thư mục cache (`~/.gensim-data/`) để tái sử dụng nhanh hơn trong các lần sau.
---

## 1.1. Task 2: 

### **Nội dung thực hiện**

1. **Tạo file:**  
   `src/representations/word_embedder.py`  
   Đây là nơi định nghĩa lớp `WordEmbedder`, phục vụ việc truy xuất và phân tích vector từ.

2. **Xây dựng lớp `WordEmbedder`:**  
   Lớp này được thiết kế như một module trung gian để thao tác với mô hình embedding GloVe.  
   Nó bao gồm ba nhóm chức năng chính:

   #### a. **Khởi tạo mô hình (`__init__`)**
   - Phương thức khởi tạo nhận tham số `model_name` (ví dụ: `'glove-wiki-gigaword-50'`).  
   - Dùng `gensim.downloader.load()` để tải mô hình tương ứng và lưu trữ trong thuộc tính nội bộ

   #### b. **Truy xuất vector (`get_vector`)**
   - Trả về **embedding vector** của một từ bất kỳ trong từ vựng.  
   - Trường hợp từ không tồn tại trong từ điển của mô hình (OOV – *Out-of-Vocabulary*), phương thức xử lý lỗi bằng cách trả về `None` hoặc vector rỗng.

   #### c. **Tính độ tương đồng (`get_similarity`)**
   - Tính **cosine similarity** giữa hai vector từ, nhằm đánh giá mức độ gần gũi về mặt ngữ nghĩa.

   #### d. **Tìm từ tương tự nhất (`get_most_similar`)**
   - Sử dụng phương thức `most_similar()` tích hợp trong gensim để truy xuất **Top N từ gần nhất** với một từ cho trước.  
   - Kết quả gồm danh sách các cặp *(từ, điểm tương đồng)*, trong đó điểm thể hiện mức độ gần về ngữ nghĩa.  

---


# Task 3: Document Embedding



## **Phương pháp**
1. Tách câu thành token bằng `Tokenizer`.  
2. Với mỗi từ, lấy vector tương ứng bằng `WordEmbedder`.  
3. Bỏ qua các từ ngoài từ vựng (OOV).  
4. Nếu không có từ hợp lệ → trả về vector 0.  
5. Nếu có → tính trung bình các vector để được document embedding.

---

## **Thử nghiệm**
Thực hiện trong `test/lab4_test.py`:
- Lấy vector của từ **“king”**  
- Tính **similarity** giữa:
  - `king` – `queen`
  - `king` – `man`
- Lấy **10 từ gần nhất với “computer”**
- Tạo document embedding cho câu: *“The queen rules the country.”*

---

## **Kết quả**
| Thao tác | Kết quả tóm tắt |
|-----------|----------------|
| `similarity(king, queen)` | ≈ 0.73 |
| `similarity(king, man)` | ≈ 0.45 |
| Từ gần “computer” | machine, software, pc, system,... |
| Document embedding | Vector 50 chiều trung bình |

---
## 2. HƯỚNG DẪN CHẠY CODE

### 2.1. Setup
```bash
cd d:\NLP\lab
pip install -r requirements.txt
```

### 2.2. Chạy Evaluation Test
```bash
python test/lab4_test.py
```
===
### 2.3. Bonus Task: Training Word2Vec
```bash
python test/lab4_embedding_training_demo.py
```
- Save model vào `results/word2vec_ewt.model`

### 2.4. Advanced Task: Apache Spark
```bash
python test/lab4_spark_word2vec_demo.py
```
- Train trên C4 corpus

---

# 3. Phân tích kết quả của các hệ thống nhúng từ và ngữ cảnh văn bản

### 3.1. Đánh giá mối liên hệ ngữ nghĩa và cấu trúc giữa các từ

#### **Thử nghiệm A: Vector hóa từ 'king'**
- Định dạng kết quả: (50,) – chính xác với cấu hình không gian 50 chiều.
- Phân bố giá trị: Các thành phần vector dao động trong khoảng [-1, 1], biểu thị việc chúng đã được chuẩn hóa.
- Đặc điểm: Đây là một dạng biểu diễn phân tán; không thể gán ý nghĩa cụ thể cho từng chiều riêng lẻ.

#### **Thử nghiệm B: Đo lường độ gần gũi ngữ nghĩa giữa các cặp từ**

| Cặp từ | Giá trị tương đồng | Nhận xét chi tiết |
|--------|--------------------|--------------------|
| king - queen | 0.7839 | Mức độ tương đồng rất cao, phản ánh mối quan hệ chặt chẽ trong cùng một phạm trù (hoàng gia) và sự xuất hiện thường xuyên trong các bối cảnh ngữ pháp tương tự. |
| king - man | 0.5309 | Mức độ tương đồng trung bình, thể hiện mối liên kết phân cấp (vua là một người đàn ông) nhưng có sự khác biệt rõ rệt về môi trường sử dụng từ. |

**Tổng kết:** Mô hình đã chứng tỏ khả năng nhận diện xuất sắc cả các mối quan hệ thay thế (paradigmatic) và quan hệ theo cấu trúc câu (syntagmatic) giữa các từ.

#### **Thử nghiệm C: Các từ có quan hệ gần gũi nhất với 'computer'**

| Vị trí | Từ | Điểm liên hệ | 
|--------|------|-------------|
| 1 | computers | 0.9165 | 
| 2 | software | 0.8815 | 
| 3 | technology | 0.8526 |
| 4 | electronic | 0.8126 | 
| 5 | internet | 0.8060 | 
| 6 | computing | 0.8026 | 
| 7 | devices | 0.8016 | 
| 8 | digital | 0.7992 |
| 9 | applications | 0.7913 | 
| 10 | pc | 0.7883 | 


### 3.2. Kiểm tra hiệu quả của phương pháp nhúng tài liệu

**Văn bản đầu vào:** "The queen rules the country."

**Phân tích chi tiết:**
- Các đơn vị xử lý (tokens): ['the', 'queen', 'rules', 'the', 'country', '.']
- Tỷ lệ từ được nhận diện (có trong từ vựng): 4/6 (tức 66.7% nội dung được bao phủ).
- Các thành phần không có trong từ vựng (OOV): ['.'] – các dấu câu thường bị loại bỏ.

**Diễn giải:**
- Vector đại diện cho toàn bộ tài liệu được xây dựng bằng cách tính trung bình các vector từ của các từ hợp lệ.
- Vector này thành công trong việc khái quát hóa các chủ đề chính như vương quyền, quản lý nhà nước và vai trò lãnh đạo.
- Kích thước: (50,), duy trì sự đồng nhất với kích thước của các vector từ riêng lẻ.

**Những điểm hạn chế:**
- Thiếu cơ chế gán trọng số cho từng từ (các từ chức năng có giá trị tương đương các từ nội dung).
- Bỏ qua thông tin về thứ tự từ (phương pháp "túi từ").
- Một vector đơn lẻ có thể chưa đủ để diễn tả đầy đủ ý nghĩa phức tạp của một tài liệu.

**Kết luận:** Mặc dù phương pháp tính trung bình đơn giản cho thấy hiệu quả tốt với các đoạn văn ngắn, các tài liệu có độ dài lớn hơn sẽ yêu cầu những phương pháp nhúng tinh vi hơn để nắm bắt ngữ nghĩa một cách toàn diện.


### 3.3. Đối chiếu giữa mô hình nhúng từ đã được huấn luyện sẵn và mô hình tự huấn luyện

| **Tiêu chí** | **Mô hình huấn luyện sẵn (GloVe)** | **Mô hình tự huấn luyện (Word2Vec)** |
|---------------|------------------------------------|--------------------------------------|
| **Bộ dữ liệu gốc để huấn luyện** | Wikipedia + Gigaword (≈ 6 tỷ từ) | UD_English-EWT (≈ 254 nghìn từ) |
| **Quy mô từ vựng** | ≈ 400.000 từ | ≈ 5.000 từ |
| **Thời gian huấn luyện** | Vài ngày (trên hệ thống mạnh) | Vài phút (CPU bình thường) |
| **Tỷ lệ từ không có trong từ vựng (OOV)** | Khoảng 5% | 30–40% |
| **Chất lượng vector biểu diễn** | Vector ổn định, thể hiện quan hệ ngữ nghĩa tốt | Vector phụ thuộc dữ liệu nhỏ nên ít khái quát hơn |
| **Khả năng thích ứng với miền dữ liệu chuyên biệt** | Hạn chế — vì mô hình học từ dữ liệu tổng quát | Rất tốt — vì có thể học đặc trưng riêng của miền dữ liệu cụ thể |
| **Kích thước vector phổ biến** | 50, 100, 200, 300 chiều | Tùy chọn linh hoạt (thường 100–200 chiều) |
| **Các trường hợp ứng dụng tối ưu** | Các tác vụ NLP phổ quát như sentiment analysis, translation, QA | Các lĩnh vực chuyên biệt hoặc dự án nhỏ, dữ liệu hẹp (ví dụ: xử lý văn bản y khoa, pháp lý) |
| **Tính khả chuyển (transferability)** | Rất cao, dễ tái sử dụng | Thấp hơn, vì phụ thuộc ngữ liệu huấn luyện |
| **Hiệu suất tính toán** | Không cần huấn luyện lại, load nhanh | Cần thời gian huấn luyện nhưng tối ưu hóa được cho bài toán cụ thể |

---

###  Nhận xét tổng quát

- **GloVe** mạnh ở **tính khái quát** và **độ ổn định** của vector biểu diễn.  
- **Word2Vec tự huấn luyện** phù hợp cho **bài toán hẹp**, nơi cần vector phản ánh đặc trưng riêng của ngữ liệu (ví dụ: dataset chuyên ngành).  

---

## 4. KHÓ KHĂN VÀ GIẢI PHÁP
### 1. Lỗi Từ OOV (Out-of-Vocabulary)
*   **Vấn đề:** Truy cập vector từ không có trong mô hình gây lỗi (`KeyError`).
*   **Giải pháp:**
    *   Kiểm tra `word in self.model.key_to_vectors` trước khi truy cập.
    *   Trả về `None` hoặc `np.zeros(self.model.vector_size)` nếu từ OOV.


### 2. Kết quả Embeddings không như mong đợi
*   **Vấn đề:** Độ tương đồng/vector không hợp lý.
*   **Giải pháp:**
    *   **Chính tả & Chữ hoa/thường:** Đảm bảo từ đầu vào chính xác và đã được chuyển đổi sang chữ thường (`word.lower()`).
    *   **Tiền xử lý:** Đảm bảo tiền xử lý văn bản nhất quán (tokenization, loại bỏ dấu câu).



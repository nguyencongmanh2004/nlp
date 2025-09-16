# Báo cáo Lab 2: Count Vectorization

---

## I. Mô tả công việc

Trong Lab 2, em đã triển khai `CountVectorizer`, một thành phần cốt lõi để chuyển đổi văn bản từ dạng chữ thành dạng vector số học. Kỹ thuật này, còn được gọi là mô hình **Túi từ (Bag-of-Words)**, là nền tảng để sử dụng dữ liệu văn bản trong các mô hình máy học.

Công việc trong lab này tái sử dụng `Tokenizer` đã được xây dựng ở Lab 1.

### 1. Thiết kế Interface `Vectorizer`

Một interface trừu tượng mới đã được định nghĩa trong `src/core/interfaces.py` để tạo ra một cấu trúc chuẩn cho tất cả các vectorizer sau này.

-   **Lớp**: `Vectorizer` (Abstract Base Class)
-   **Các phương thức**:
    1.  `fit(self, corpus: list[str])`: "Học" bộ từ vựng từ một danh sách các văn bản (corpus).
    2.  `transform(self, documents: list[str]) -> list[list[int]]`: Chuyển đổi một danh sách các văn bản thành một danh sách các vector đếm, dựa trên bộ từ vựng đã học.
    3.  `fit_transform(self, corpus: list[str]) -> list[list[int]]`: Một phương thức tiện ích, thực hiện cả `fit` và `transform` trên cùng một dữ liệu.

### 2. Cài đặt `CountVectorizer`

Đây là lớp triển khai cụ thể của mô hình Bag-of-Words.

-   **File**: `src/representations/count_vectorizer.py`
-   **Kế thừa**: Lớp `CountVectorizer` kế thừa từ interface `Vectorizer`.
-   **Thuộc tính**:
    -   `tokenizer`: Một instance của `Tokenizer` (ví dụ: `RegexTokenizer` từ Lab 1) được truyền vào qua constructor.
    -   `vocabulary_`: Một dictionary (`dict[str, int]`) để lưu trữ ánh xạ từ mỗi từ (token) sang một chỉ số (index) duy nhất.
-   **Logic hoạt động của các phương thức**:
    -   **`fit(corpus)`**:
        1.  Duyệt qua từng văn bản trong `corpus`.
        2.  Sử dụng `tokenizer` để tách mỗi văn bản thành một danh sách các token.
        3.  Tập hợp tất cả các token duy nhất từ toàn bộ corpus vào một tập hợp (set).
        4.  Sắp xếp các token duy nhất này theo thứ tự bảng chữ cái và tạo ra `vocabulary_`, trong đó mỗi token được gán một index từ `0` đến `N-1` (với `N` là tổng số token duy nhất).
    -   **`transform(documents)`**:
        1.  Đối với mỗi văn bản trong `documents`:
        2.  Tạo một vector chứa toàn số `0`, có độ dài bằng kích thước của `vocabulary_`.
        3.  Tách văn bản thành các token.
        4.  Với mỗi token, nếu nó tồn tại trong `vocabulary_`, tìm index tương ứng và tăng giá trị (số đếm) tại vị trí index đó trong vector lên `1`.
        5.  Trả về danh sách các vector đã được tạo.

---

## II. Kết quả chạy code

Dưới đây là kết quả khi chạy `CountVectorizer` với `RegexTokenizer` trên corpus mẫu.

### 1. Corpus mẫu

```python
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
Vocabulary = {
    '.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 
    'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9
}
Document-Term Matrix
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

```

## III.Giải thích kết quả

Vocabulary: tập hợp tất cả các token duy nhất xuất hiện trong corpus/dataset.

Document-Term Matrix:

Mỗi hàng tương ứng một văn bản.

Mỗi cột tương ứng một token.

Giá trị là số lần token xuất hiện trong văn bản đó.

CountVectorizer: biến đổi văn bản thành vector số, làm đầu vào cho các mô hình ML như Naive Bayes, Logistic Regression, SVM…

Khó khăn: khi corpus lớn, vocabulary cũng lớn, cần xử lý ma trận thưa (sparse matrix) để tiết kiệm bộ nhớ.
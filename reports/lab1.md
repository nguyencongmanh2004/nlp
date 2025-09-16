

# Báo cáo Lab 1: Text Tokenization

---

## I. Mô tả công việc

Trong Lab 1, em đã triển khai các thành phần cơ bản để thực hiện **tokenization** (tách từ), một bước tiền xử lý nền tảng trong Xử lý Ngôn ngữ Tự nhiên (NLP).

### 1. Thiết kế Interface `Tokenizer`

Để đảm bảo tính module hóa và dễ mở rộng, một interface trừu tượng đã được định nghĩa:

-   **File**: `src/core/interfaces.py`
-   **Lớp**: `Tokenizer` (Abstract Base Class)
-   **Phương thức**: Định nghĩa một phương thức trừu tượng duy nhất là `tokenize(self, text: str) -> list[str]`. Điều này bắt buộc tất cả các lớp tokenizer kế thừa phải triển khai logic tách từ của riêng mình, trong khi vẫn tuân thủ một cấu trúc chung.

### 2. Cài đặt `SimpleTokenizer`

Đây là một tokenizer cơ bản, xử lý văn bản bằng các quy tắc đơn giản.

-   **File**: `src/preprocessing/simple_tokenizer.py`
-   **Logic hoạt động**:
    1.  **Chuẩn hóa chữ thường**: Toàn bộ văn bản đầu vào được chuyển thành chữ thường bằng phương thức `.lower()`.
    2.  **Xử lý dấu câu**: Các dấu câu cơ bản (`.`, `,`, `?`, `!`) được tách ra khỏi từ liền kề bằng cách thêm một khoảng trắng vào trước và sau chúng.
    3.  **Tách từ**: Văn bản sau khi xử lý được tách thành danh sách các token bằng phương thức `.split()` dựa trên khoảng trắng.

### 3. Cài đặt `RegexTokenizer` (Bonus)

Đây là một tokenizer nâng cao hơn, sử dụng sức mạnh của Biểu thức chính quy (Regular Expressions) để có kết quả tách từ tốt hơn.

-   **File**: `src/preprocessing/regex_tokenizer.py`
-   **Logic hoạt động**:
    -   Sử dụng một biểu thức chính quy duy nhất để tìm tất cả các chuỗi con khớp với mẫu token:
      ```regex
      \w+|[^\w\s]
      ```
    -   **Giải thích Regex**:
        -   `\w+`: Tìm và khớp với một hoặc nhiều ký tự "word" (chữ cái `a-z`, `A-Z`; số `0-9`; và dấu gạch dưới `_`).
        -   `|`: Toán tử "HOẶC".
        -   `[^\w\s]`: Tìm và khớp với một ký tự bất kỳ **không phải** là ký tự "word" và cũng **không phải** là khoảng trắng. Điều này giúp tách riêng các dấu câu và ký hiệu đặc biệt (`-`, `=`, `#`, `!`, `@`, ...).

---

## II. Kết quả chạy code

### 1. Output trên các câu mẫu

```plaintext
--- Testing SimpleTokenizer and RegexTokenizer ---

Original: "Hello, world! This is a test."
SimpleTokenizer: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer:  ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

---
Original: "NLP is fascinating... isn't it?"
SimpleTokenizer: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', '?']
RegexTokenizer:  ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', '?']

---
Original: "Let's see how it handles 123 numbers and punctuation!"
SimpleTokenizer: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer:  ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
2. Output trên dataset UD_English-EWT

2.2. Ví dụ dataset UD_English-EWT (20 tokens đầu)
SimpleTokenizer =>
['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani,', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim,', 'near', 'the']

RegexTokenizer =>
['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']

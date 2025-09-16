
Báo cáo Lab 1: Text Tokenization
I. Mô tả công việc

Trong Lab 1, mục tiêu chính là tìm hiểu và triển khai bước tiền xử lý cơ bản và quan trọng trong NLP: tách từ (tokenization). Công việc được chia thành các phần chính sau:

Thiết kế Interface Tokenizer:

Để đảm bảo tính nhất quán và khả năng mở rộng, em đã tạo một lớp trừu tượng (abstract base class) có tên Tokenizer trong file src/core/interfaces.py.

Interface này định nghĩa một phương thức bắt buộc là tokenize(self, text: str) -> list[str]. Mọi lớp tokenizer được triển khai sau này đều phải kế thừa từ interface này và cài đặt phương thức tokenize, giúp cho cấu trúc của dự án rõ ràng và dễ bảo trì.

Cài đặt SimpleTokenizer:

Em đã tạo lớp SimpleTokenizer trong file src/preprocessing/simple_tokenizer.py, kế thừa từ Tokenizer.

Phương pháp xử lý:

Bước 1: Chuyển sang chữ thường: Toàn bộ văn bản đầu vào được chuyển đổi thành chữ thường bằng phương thức .lower() để đảm bảo tính đồng nhất (ví dụ: "Hello" và "hello" được coi là một).

Bước 2: Xử lý dấu câu: Để tách các dấu câu cơ bản (., ,, ?, !) ra khỏi các từ, em đã thêm một khoảng trắng vào trước và sau mỗi dấu câu này trong văn bản.

Bước 3: Tách từ: Cuối cùng, văn bản đã được xử lý sẽ được tách thành một danh sách các token dựa trên các khoảng trắng bằng phương thức .split().

Cài đặt RegexTokenizer (Bonus):

Để có một giải pháp mạnh mẽ và linh hoạt hơn, em đã tạo lớp RegexTokenizer trong file src/preprocessing/regex_tokenizer.py.

Phương pháp xử lý:

Lớp này sử dụng một biểu thức chính quy (regular expression) duy nhất để tìm tất cả các token trong văn bản: \w+|[^\w\s].

Giải thích Regex:

\w+: Khớp với một hoặc nhiều ký tự "word" liên tiếp (bao gồm chữ cái, số, và dấu gạch dưới _).

|: Toán tử "HOẶC" (OR).

[^\w\s]: Khớp với một ký tự đơn lẻ bất kỳ không phải là ký tự "word" (\w) và cũng không phải là ký tự khoảng trắng (\s). Điều này giúp tách riêng các dấu câu và các ký hiệu đặc biệt khác (-, =, #, ...).

II. Kết quả chạy code
1. Output trên các câu mẫu

Dưới đây là kết quả khi chạy hai tokenizer trên các câu văn bản mẫu:

code
Code
download
content_copy
expand_less

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
RegexTokenizer:  ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']```

### **2. Output trên dataset UD_English-EWT**

Kết quả khi áp dụng tokenizer trên 500 ký tự đầu tiên của tập dữ liệu `UD_English-EWT`:

```plaintext
--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: # newdoc id = weblog-XML_BLOG_00001_20040301-0001
# sent_id = weblog-XML_BLOG_00001_20040301-0001_1...

SimpleTokenizer Output (first 20 tokens): 
['#', 'newdoc', 'id', '=', 'weblog-xml_blog_00001_20040301-0001', '#', 'sent_id', '=', 'weblog-xml_blog_00001_20040301-0001_1', '#', 'text', '=', 'from', 'the', 'exchange', 'in', 'honolulu', ',', 'to', 'the']

RegexTokenizer Output (first 20 tokens): 
['#', 'newdoc', 'id', '=', 'weblog', '-', 'xml_blog_00001_20040301', '-', '0001', '#', 'sent_id', '=', 'weblog', '-', 'xml_blog_00001_20040301', '-', '0001', '_', '1', '#']
III. Giải thích kết quả và phân tích
1. So sánh SimpleTokenizer và RegexTokenizer

Trên các câu mẫu đơn giản: Cả hai tokenizer đều cho ra kết quả gần như giống hệt nhau. SimpleTokenizer hoạt động tốt vì các câu này chỉ chứa các dấu câu cơ bản đã được xử lý trước.

Trên dữ liệu thực tế (UD_English-EWT): Sự khác biệt trở nên rõ rệt:

SimpleTokenizer: Đã thất bại trong việc xử lý các chuỗi phức tạp như weblog-xml_blog_00001_20040301-0001. Vì nó chỉ tách dựa trên khoảng trắng nên toàn bộ chuỗi này được coi là một token duy nhất. Điều này là không chính xác vì chuỗi này chứa nhiều thông tin có thể tách rời (từ, số, dấu gạch ngang).

RegexTokenizer: Thể hiện sự vượt trội rõ ràng. Nó đã phân tách chuỗi trên một cách thông minh thành ['weblog', '-', 'xml_blog_00001_20040301', '-', '0001']. Biểu thức chính quy đã nhận diện được các chuỗi chữ và số (\w+) và các ký tự đặc biệt ([^\w\s]) như các token riêng biệt. Kết quả này chi tiết và hữu ích hơn nhiều cho các bước xử lý sau này.

2. Khó khăn gặp phải và cách giải quyết

Khó khăn: Thách thức chính với SimpleTokenizer là làm thế nào để tách các dấu câu dính liền với từ (ví dụ: world!) mà không dùng đến regex. Nếu chỉ dùng text.split() thì "world!" sẽ là một token.

Giải pháp: Em đã giải quyết vấn đề này bằng một bước tiền xử lý đơn giản: thay thế mỗi dấu câu p trong danh sách ['.', ',', '?', '!'] bằng " " + p + " ". Điều này đảm bảo luôn có khoảng trắng xung quanh các dấu câu, giúp phương thức .split() hoạt động chính xác.

Nhận xét: Mặc dù SimpleTokenizer dễ cài đặt, nó thiếu đi sự linh hoạt và dễ bị "bẻ gãy" bởi các trường hợp không lường trước. RegexTokenizer, tuy phức tạp hơn để thiết kế ban đầu, nhưng lại cung cấp một giải pháp mạnh mẽ và tổng quát hơn cho bài toán tách từ trong thực tế.
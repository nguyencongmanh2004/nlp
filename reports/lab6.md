# BÁO CÁO LAB 6: GIỚI THIỆU VỀ TRANSFORMERS

**Mã số sinh viên / Họ tên:** Nguyễn Công Mạnh --- 22001272

## I. CÁC BƯỚC THỰC HIỆN VÀ KẾT QUẢ

Trong bài lab này, chúng ta sử dụng pipeline của Hugging Face để thực
hiện các tác vụ NLP cơ bản.

------------------------------------------------------------------------

# **Bài 1: Khôi phục Masked Token (Masked Language Modeling)**

### **Mục đích**

Sử dụng mô hình Encoder-only để dự đoán từ bị thiếu trong câu.\
Mô hình: `distilbert/distilroberta-base`\
Câu đầu vào: `ha noi is <mask> of vietnam`

------------------------------------------------------------------------

## **1. Các bước thực hiện**

Sử dụng `pipeline("fill-mask")` với mô hình và tokenizer từ Hugging
Face.

``` python
from transformers import pipeline, AutoTokenizer

model_name = "distilbert/distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
mask_token = tokenizer.mask_token  # "<mask>"

mask_filler = pipeline('fill-mask', model=model_name, tokenizer=tokenizer)

input_sentence = f"ha noi is {mask_token} of vietnam"
predictions = mask_filler(input_sentence, top_k=5)
```

------------------------------------------------------------------------

## **2. Kết quả**

  Rank   Predicted Token   Confidence   Full Sentence
  ------ ----------------- ------------ -----------------------------
  1      king              0.2076       ha noi is king of vietnam
  2      lord              0.0848       ha noi is lord of vietnam
  3      part              0.0474       ha noi is part of vietnam
  4      afraid            0.0324       ha noi is afraid of vietnam
  5      worthy            0.0290       ha noi is worthy of vietnam

### **Nhận xét**

Các từ được dự đoán như **king**, **lord**, **part** cho thấy mô hình
hiểu ngữ cảnh nhưng **không nắm được kiến thức thực tế** rằng Hà Nội là
thủ đô (**capital**) của Việt Nam.

------------------------------------------------------------------------

## **3. Tại sao các mô hình Encoder-only như BERT phù hợp cho tác vụ này?**

### ** 1. Hiểu ngữ cảnh hai chiều (Bidirectional Context)**

Encoder của Transformer sử dụng **self-attention hai chiều**, cho phép
mô hình nhìn cả **bên trái** và **bên phải** của từ bị che `<mask>` cùng
lúc.

→ Rất phù hợp cho Masked Language Modeling.

### ** 2. Thiết kế tối ưu cho các tác vụ hiểu ngôn ngữ**

Encoder-only phù hợp với: - Khôi phục Masked Token (MLM) - Phân loại văn
bản - NER - Trả lời câu hỏi (QA)

### ** 3. Không phù hợp cho sinh văn bản**

Encoder-only không dự đoán token tiếp theo nên **không dùng cho text
generation** --- đó là nhiệm vụ của mô hình decoder-only (GPT).
# Bài 2: Sinh văn bản (Text Generation)

## Mục đích

Sử dụng mô hình **Decoder-only** để sinh ra phần tiếp theo của một câu
mồi (prompt).\
Mô hình sử dụng: Mặc định của `pipeline('text-generation')` (thường là
GPT-2).\
Câu mồi:

    the best thing about learning NLP is

## 1. Các bước thực hiện

Sử dụng `pipeline("text-generation")` với: - `max_length = 50` -
`num_return_sequences = 1`

### Python Code

``` python
from transformers import pipeline

generator = pipeline('text-generation')  # Sử dụng mô hình mặc định (GPT-2)

prompt = "the best thing about learning NLP is"
generated_texts = generator(prompt, max_length=50, num_return_sequences=1)

# In kết quả
print(generated_texts)
```

## 2. Kết quả

Mô hình đã sinh ra đoạn văn bản sau (bắt đầu bằng câu mồi):

    the best thing about learning NLP is that you have a lot of freedom to do what you want with your time. 
    It makes it easy to do things that you're not allowed to do."  
    He added that by having a "comfortable space" in the classroom, students can become better communicators. 
    "It's a positive thing for all of us because it's very rewarding," he said  
    "We can also say, 'OK, you're learning to make it better. You have to be a better communicator, as opposed to the one you are."

## 2.1. Kết quả sinh ra có hợp lý không?

-   Câu đầu tiên tạo ra khá hợp lý\
    → *"you have a lot of freedom to do what you want with your time"*

-   Tuy nhiên mô hình **nhanh chóng bị lạc đề**, nói sang:

    -   "comfortable space"
    -   "students becoming better communicators"
    -   nội dung mang tính nghiên cứu -- **không liên quan đến NLP**

Điều này thường xảy ra với **GPT-2 chưa tinh chỉnh**, do mô hình chỉ học
phân phối ngôn ngữ chung.

## 2.2. Tại sao các mô hình Decoder-only (như GPT) phù hợp cho tác vụ sinh văn bản?

### Tối ưu cho Next Token Prediction

GPT được huấn luyện để dự đoán **token tiếp theo** trong chuỗi\
→ giúp mô hình sinh văn bản trôi chảy theo hướng **một chiều
(unidirectional)**.

###  Kiến trúc Autoregressive

Mỗi token **chỉ nhìn thấy token phía trước**, giúp văn bản sinh ra tự
nhiên và mạch lạc theo thời gian.

### Hoạt động tốt với các tác vụ:

-   Text Generation
-   Story/Paragraph Completion
-   Chatbots
-   Summarization dạng sinh (generative)

# Bài 3: Tính toán Vector biểu diễn của câu (Sentence Representation)

**Mục đích:** Sử dụng mô hình Encoder-only (BERT) để tạo ra vector biểu diễn ngữ nghĩa cho toàn bộ câu (Sentence Embedding) bằng phương pháp Mean Pooling.  
**Mô hình sử dụng:** `bert-base-uncased`  
**Câu đầu vào:** `this is sample sentence`

---

## 1. Các bước thực hiện

1. Tải `AutoTokenizer` và `AutoModel` cho `bert-base-uncased`.
2. Tokenize câu đầu vào và tạo `input_ids`, `attention_mask`.
3. Đưa inputs vào mô hình để lấy `last_hidden_state`.
4. Áp dụng **Mean Pooling** (trung bình hóa các vector hidden state, có tính đến `attention_mask` để bỏ qua token đệm) để tính ra vector biểu diễn câu.

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = ['this is sample sentence']
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

# Mean Pooling
attention_mask = inputs['attention_mask']
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
sentence_embedding = sum_embeddings / sum_mask

print("Kích thước của vector:", sentence_embedding.shape)
# ... in vector kết quả
```

## 2. Kết quả

- Vector biểu diễn của câu: (Một tensor dài 768 chiều)

### 2.1. Kích thước (chiều) của vector biểu diễn

- Kích thước của vector là **768 chiều**.  
- Con số này tương ứng với **Hidden Size (H)** (`hidden_size` trong cấu hình của mô hình) của `bert-base-uncased`.  
- Đây là số chiều của mỗi vector token đầu ra (`last_hidden_state`) và cũng là số chiều của vector biểu diễn câu sau khi thực hiện Mean Pooling.

### 2.2. Tại sao phải sử dụng `attention_mask` khi tính toán Mean Pooling?

- `attention_mask` được sử dụng để bỏ qua các token đệm (`padding tokens`) trong quá trình tính trung bình.  
- Khi xử lý theo batch, các câu ngắn hơn được thêm token đệm (`[PAD]`) để có cùng độ dài.  
- Nếu tính trung bình trên tất cả các token (bao gồm cả `[PAD]`), vector biểu diễn câu sẽ bị sai lệch (bị kéo về giá trị trung bình của vector đệm), không phản ánh đúng ngữ nghĩa của câu gốc.

- Ví dụ kích thước của vector: `torch.Size([1, 768])`.

---

## III. Khó khăn và giải pháp

| Khó khăn                                         | Mô tả & Giải pháp                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dự đoán không chính xác về mặt kiến thức (Bài 1) | Trong Bài 1, mô hình `distilbert/distilroberta-base` đưa ra kết quả cao nhất là `king`, `lord` thay vì từ chính xác là `capital` cho câu `"Hanoi is <mask> of Vietnam"`. **Giải pháp:** Sử dụng các mô hình đã được tinh chỉnh (fine-tuned) trên tập dữ liệu đa ngôn ngữ lớn hơn hoặc các mô hình tập trung vào kiến thức (knowledge-enhanced models).                                                             |
| Văn bản sinh ra bị lạc đề (Bài 2)                | Mô hình GPT-2 ban đầu tạo ra ý hợp lý (`freedom to do what you want`) nhưng sau đó nhanh chóng bị lạc đề và bắt đầu nói về một nghiên cứu của UC Berkeley. **Giải pháp:** Tinh chỉnh mô hình trên tập dữ liệu chuyên biệt về NLP/Machine Learning, hoặc điều chỉnh các tham số sinh văn bản như `temperature` hoặc `top_k/top_p` để kiểm soát tính ngẫu nhiên và độ tập trung của văn bản. |
| Hiểu và thực hiện Mean Pooling (Bài 3)           | Việc tính toán vector câu bằng cách lấy trung bình (Mean Pooling) các vector token đòi hỏi phải xử lý đúng token đệm (padding) để tránh sai lệch kết quả. **Giải pháp:** Sử dụng `attention_mask` để nhân (element-wise multiplication) với `last_hidden_state`, sau đó chỉ tính tổng/trung bình trên các token thực.                                                                                                  |

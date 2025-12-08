
# BÁO CÁO THỰC HÀNH LAB 6: PHÂN TÍCH CÚ PHÁP PHỤ THUỘC (DEPENDENCY PARSING)

**Họ và tên:** Nguyễn Công Mạnh

## 1. Mục tiêu

* Sử dụng thư viện `spaCy` để phân tích cấu trúc ngữ pháp câu (Dependency Parsing).
* Trực quan hóa cây phụ thuộc.
* Duyệt cây (Tree Traversal) để trích xuất thông tin (Chủ ngữ, Tân ngữ, Động từ chính, Cụm danh từ).

---

## 2. Nội dung thực hiện và Kết quả

### Phần 1: Cài đặt và Khởi tạo

Tôi đã tiến hành cài đặt thư viện và tải mô hình ngôn ngữ mức trung bình (`md` - medium) để đảm bảo có đủ thông tin vector và cú pháp.

**Code thực hiện:**

```bash
!pip install -U spacy
!python -m spacy download en_core_web_md
```

**Kết quả:** Cài đặt thành công, các thư viện phụ thuộc đã thỏa mãn.

### Phần 2: Phân tích câu và Trực quan hóa

**Mục tiêu:** Xác định quan hệ giữa các từ trong câu *"The quick brown fox jumps over the lazy dog."*

**Code thực hiện:**

```python
import spacy
from spacy import displacy

# Tải mô hình
nlp = spacy.load("en_core_web_md")

# Phân tích câu
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# Hiển thị cây (Lưu ý: Trong Notebook nên dùng render thay vì serve)
# displacy.serve(doc, style='dep')
```

**Trả lời câu hỏi từ kết quả phân tích:**

1. **Từ gốc (ROOT):** `jumps`.
2. **Từ phụ thuộc của "jumps":** `fox` (chủ ngữ) và `over` (giới từ).
3. **"Fox" là Head của:** `The` (det), `quick` (amod), `brown` (amod).

### Phần 3: Truy cập thuộc tính Token

**Mục tiêu:** Hiển thị chi tiết nhãn quan hệ (Dependency labels) và từ cha (Head) của câu *"Apple is looking at buying U.K. startup for $1 billion"*.

**Code thực hiện:**

```python
text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

print(f"{'TEXT':<12} | {'DEP':<10} | {'HEAD TEXT':<12} | {'HEAD POS':<8} | {'CHILDREN'}")
print("-" * 70)

for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:<12} | {token.dep_:<10} | {token.head.text:<12} | {token.head.pos_:<8} | {children}")
```

**Kết quả trích xuất (Một phần):**

* **Apple**: `nsubj` (chủ ngữ) của *looking*.
* **looking**: `ROOT` (động từ chính) của câu.
* **U.K.**: `compound` (danh từ ghép) bổ nghĩa cho *startup*.

### Phần 4: Trích xuất thông tin (Information Extraction)

#### 4.1. Tìm bộ ba (Chủ ngữ - Động từ - Tân ngữ)

**Code thực hiện:**

```python
text = "the cat chased the mouse and the dog watched them"
doc = nlp(text)

for token in doc:
    if token.pos_ == "VERB":
        verb = token.text
        subject = ""
        obj = ""
        # Tìm chủ ngữ và tân ngữ trong con của động từ
        for child in token.children:
            if child.dep_ == "nsubj":
                subject = child.text
            if child.dep_ == "dobj":
                obj = child.text
        
        if subject and obj:
            print(f"Found Triplet: ({subject}, {verb}, {obj})")
```

**Kết quả:**

* Found Triplet: `(cat, chased, mouse)`
* Found Triplet: `(dog, watched, them)`

#### 4.2. Tìm tính từ bổ nghĩa cho danh từ

**Code thực hiện:**

```python
text = "the big, fluffy white cat is sleeping on the warm mat."
doc = nlp(text)

for token in doc:
    if token.pos_ == "NOUN":
        adjectives = []
        for child in token.children:
            if child.dep_ == "amod":
                adjectives.append(child.text)
        if adjectives:
            print(f"Danh từ '{token.text}' được bổ nghĩa bởi các tính từ: {adjectives}")
```

**Kết quả:**

* Danh từ 'cat': `['big', 'fluffy', 'white']`
* Danh từ 'mat': `['warm']`

### Phần 5: Bài tập tự luyện (Nâng cao)

#### Bài 1: Tìm động từ chính (Main Verb)

**Code thực hiện:**

```python
def find_main_verb(doc):
    for token in doc:
        if token.dep_ == "ROOT":
            return token.text
    return None

sentence = "The manager decided to cancel the meeting because it was..."
doc = nlp(sentence)
print(f"Main Verb: {find_main_verb(doc)}")
```

**Kết quả:** Main Verb: `decided`.

#### Bài 2: Trích xuất cụm danh từ (Noun Chunks) thủ công

Phần này yêu cầu xử lý logic để tránh lấy trùng lặp các danh từ ghép (như "U.K." trong "U.K. startup").

**Code thực hiện:**

```python
def get_noun_chunks_manual(doc):
    print(f"{'HEAD NOUN':<15} {'DETECTED CHUNK'}")
    print("-" * 40)
    for token in doc:
        # Chỉ xét NOUN hoặc PROPN
        if token.pos_ in ["NOUN", "PROPN"]:
            # Bỏ qua nếu là một phần của danh từ ghép (để tránh trùng lặp)
            if token.dep_ == "compound":
                continue
            # Lấy toàn bộ cây con (subtree) của danh từ đó
            chunk_words = [t.text for t in token.subtree]
            chunk_text = " ".join(chunk_words)
            print(f"{token.text:<15} {chunk_text}")

# So sánh với spaCy
sentence = "The manager decided to cancel the meeting..."
doc = nlp(sentence)
get_noun_chunks_manual(doc)
```

**Kết quả:** Code thủ công trích xuất chính xác: `The manager`, `the meeting`... tương đương với kết quả tích hợp sẵn của spaCy.

#### Bài 3: Tìm đường đi ngắn nhất tới ROOT

Đây là bài toán duyệt ngược cây (Bottom-up traversal).

**Code thực hiện:**

```python
def get_path_to_root(token):
    path = []
    # Duyệt ngược lên cha cho đến khi gặp ROOT (nơi head == chính nó)
    while token.head != token:
        path.append(token)
        token = token.head
    path.append(token) # Thêm ROOT vào cuối
    return path

# Test với từ "lazy"
sentence = "The quick brown fox jumps over the lazy dog."
doc = nlp(sentence)
selected_token = doc[6] # "lazy"
path = get_path_to_root(selected_token)

# In kết quả
for i, token in enumerate(path):
    arrow = "-->" if i < len(path) - 1 else "(ROOT)"
    print(f"{token.text:<10} [{token.dep_}] {arrow}")
```

**Kết quả:** Đường đi tìm được (4 bước): `lazy` [amod] --> `dog` [pobj] --> `over` [prep] --> `jumps` [ROOT].

---

## 3. Khó khăn và Vấn đề gặp phải

Trong quá trình thực hiện, tôi đã ghi nhận các vấn đề sau:

1. **Lỗi hiển thị DisplaCy:**

   * **Vấn đề:** Khi chạy `displacy.serve()` trong Jupyter Notebook, hệ thống báo cảnh báo `UserWarning: [W011]`.
   * **Nguyên nhân:** `serve()` cố gắng chạy một web server mới, xung đột với môi trường Notebook hiện tại.
   * **Khắc phục:** Thay thế bằng `displacy.render(doc, style='dep', jupyter=True)` để hiển thị trực tiếp kết quả ngay trong dòng code thay vì mở cổng localhost mới.

2. **Xử lý Danh từ ghép (Compound Nouns):**

   * **Vấn đề:** Khi trích xuất cụm danh từ (Bài tập 2), nếu chỉ lấy tất cả `NOUN`, kết quả sẽ bị dư thừa. Ví dụ "U.K. startup", cả 2 từ đều là danh từ.
   * **Giải pháp:** Tôi đã thêm điều kiện `if token.dep_ == "compound": continue` để bỏ qua danh từ phụ, chỉ bắt đầu trích xuất từ danh từ chính (Head noun).

## 4. Kết luận

Bài thực hành đã hoàn thành đầy đủ các mục tiêu đề ra. Tôi đã nắm được cách spaCy tổ chức dữ liệu dưới dạng cây, hiểu rõ vai trò của `HEAD` và `CHILDREN` trong việc xác định ý nghĩa câu. Các đoạn code tự viết (tìm đường đi, trích xuất cụm từ) đã hoạt động chính xác và khớp với kết quả lý thuyết.


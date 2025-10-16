import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

DATA_PATH = 'src/data/UD_English-EWT/en_ewt-ud-train.txt'
MODEL_PATH = "./results/word2vec_ewt.model"

def stream_sentences(file_path):
    """
    Đọc dữ liệu dạng streaming (không load hết vào RAM)
    Mỗi dòng được tách thành các token bằng gensim.simple_preprocess
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield simple_preprocess(line)

def train_word2vec(sentences):
    """
    Huấn luyện mô hình Word2Vec từ dữ liệu
    """
    print("🔹 Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,    # kích thước vector embedding
        window=5,           # ngữ cảnh 5 từ
        min_count=2,        # bỏ qua các từ xuất hiện < 2 lần
        workers=4,          # số luồng CPU
        sg=1                # dùng Skip-Gram (0 = CBOW, 1 = Skip-Gram)
    )
    print("Training completed.")
    return model

def main():
    os.makedirs("results", exist_ok=True)

    print("Reading data from:", DATA_PATH)
    sentences = list(stream_sentences(DATA_PATH))
    print(f"Loaded {len(sentences)} sentences.")

    model = train_word2vec(sentences)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 4️⃣ Dùng thử model: tìm từ tương tự và phép toán vector
    test_word = "dog"
    if test_word in model.wv:
        print(f"\nTop similar words to '{test_word}':")
        for word, score in model.wv.most_similar(test_word, topn=5):
            print(f"  {word} ({score:.3f})")

    # Phép toán vector: king - man + woman ≈ queen
    analogy = ("king", "man", "woman")
    if all(w in model.wv for w in analogy):
        result = model.wv.most_similar(positive=[analogy[0], analogy[2]], negative=[analogy[1]], topn=1)
        print(f"\nAnalogy result: {analogy[0]} - {analogy[1]} + {analogy[2]} ≈ {result[0][0]} ({result[0][1]:.3f})")

if __name__ == "__main__":
    main()

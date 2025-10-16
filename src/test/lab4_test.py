# test/lab4_test.py

from src.representations.word_embedder import WordEmbedder
import numpy as np

def main():
    print("Initializing WordEmbedder with 'glove-wiki-gigaword-50'...")
    embedder = WordEmbedder("glove-wiki-gigaword-50")

    print("\n=== Test 1: Get vector for word 'king' ===")
    king_vector = embedder.get_vector("king")
    print("Vector for 'king':", king_vector[:10], "...")  # chỉ in 10 phần tử đầu

    print("\n=== Test 2: Similarity ===")
    sim_king_queen = embedder.get_similarity("king", "queen")
    sim_king_man = embedder.get_similarity("king", "man")
    print(f"Similarity('king', 'queen') = {sim_king_queen:.4f}")
    print(f"Similarity('king', 'man')   = {sim_king_man:.4f}")

    print("\n=== Test 3: Most similar words to 'computer' ===")
    similar_to_computer = embedder.get_most_similar("computer", top_n=10)
    for word, score in similar_to_computer:
        print(f"{word:15s}  →  {score:.4f}")

    print("\n=== Test 4: Embed sentence ===")
    sentence = "The queen rules the country."
    doc_vector = embedder.embed_document(sentence)
    print("Document vector shape:", doc_vector.shape)
    print("First 10 values:", np.round(doc_vector[:10], 4))

    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()

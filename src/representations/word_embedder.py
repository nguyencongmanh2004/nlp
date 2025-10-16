import gensim.downloader as api
import numpy as np
from src.core.regex_tokenizer import RegexTokenizer
class WordEmbedder:
    def __init__(self , model_name : str = 'glove-wiki-gigaword-50'):
        self.model = api.load(model_name)
    def get_vector(self , word : str) :
        """==========
        returns the embedding vector for a given word.
        handel case where the word is not in the vocabulary.
        """
        if word in self.model.key_to_index:
            return self.model[word]
        else :
            print("Word not in vocabulary.")
            return np.zeros(self.model.vector_size)
    def get_similarity(self , word1 : str , word2 : str ):
        """return the similarity between two words"""
        if word1 not in self.model.key_to_index or word2 not in self.model.key_to_index:
            print("Word not in vocabulary.")
        return self.model.similarity(word1, word2)
    def get_most_similar(self , word : str , top_n : int ) :
        """ return top N most similar words """
        if word not in self.model.key_to_index:
            print(f"{word} not in vocabulary.")
            return []
        else :
            return self.model.most_similar(word , topn = top_n)

    def embed_document(self , document : str) :

        tokenizer = RegexTokenizer()
        tokens = tokenizer.tokenize(document)
        # get vector for each token
        vectors = []
        for token in tokens :
            vec = self.get_vector(token)
            if vec is not None :
                vectors.append(vec)
        if not vectors :
            print("No vectors found.")
            return np.zeros(self.model.vector_size)
        # doc embed is average of the words in doc
        doc_vec = np.mean(vectors, axis=0)
        return doc_vec






from src.core.interfaces import Vectorizer, Tokenizer


class CountVectorizer(Vectorizer) :
    def __init__(self , tokenizer : Tokenizer ):
        self.tokenizer = tokenizer
        self.vocabulary : dict[str , int] = {}
    def fit(self , corpus : list[str] ):
        tokens_set = set()
        for doc in corpus :
            tokens = self.tokenizer.tokenize(doc)
            tokens_set.update(tokens)
        self.vocabulary = {token : i for i , token in enumerate(tokens_set)}
    def transform(self, documents : list[str], np=None) :
        vectors = []
        for doc in documents :
            tokens = self.tokenizer.tokenize(doc)
            vector = [ 1 if i in tokens else 0 for i in self.vocabulary ]
            vectors.append(vector)
        return vectors
    def fit_transform(self , documents : list[str] , np=None) :
        self.fit(documents)
        return self.transform(documents, np)





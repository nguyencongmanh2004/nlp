from typing import Dict

from src.representations.count_vectorizer import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
class TextClassifier:
    def __init__(self  , vectorizer : CountVectorizer) :
        self.vectorizer = vectorizer
        self._model = None
    def fit(self , text : list[str] , labels : list[int]):
        """This method fits the model to the given text"""
        #vectorizer input to spare vector
        X = self.vectorizer.fit_transform(text)
        self._model = LogisticRegression(solver='liblinear')
        self._model.fit(X, labels)
    def predict(self , text  : list[str]) -> list[int] :
        X = self.vectorizer.transform(text)
        return self._model.predict(X)

    def evaluate( self , y_true : list[int] , y_pred : list[int])-> Dict[str, float] :
        """This method evaluates the model on the given text"""
        acc = accuracy_score(y_true , y_pred)
        prec = precision_score(y_true , y_pred)
        rec = recall_score(y_true , y_pred)
        f1 = f1_score(y_true , y_pred)
        return {'accuracy' : acc, 'precision' : prec, 'recall' : rec, 'f1' : f1}
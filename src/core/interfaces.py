from abc import ABC ,  abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self , text : str ) -> list[str]:
        pass
class Vectorizer(ABC):
    @abstractmethod
    def fit(self , corpus : list[str] ):
        """Learns the vocabulary from a list of documents(corpus)."""
        pass
    @abstractmethod
    def transform(self , documents : list[str]) -> list[list[int]]:
        """transform a list of documents into a list of count vectors based on the vocabulary."""
    @abstractmethod
    def fit_transform(self , documents : list[str]) -> list[list[int]]:
        """ A convenience method that performs fit and then transform on the same data."""


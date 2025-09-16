from src.core.interfaces import Tokenizer
import re

class RegexTokenizer(Tokenizer):
    def tokenize(self , text : str ) -> list[str]:
        text = text.lower()

        tokens = re.findall(r"\w+|[^\w\s]" , text )
        return tokens

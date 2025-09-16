from src.core.interfaces import Tokenizer
from src.core.regex_tokenizer import RegexTokenizer
from src.core.simple_tokenizer import SimpleTokenizer


def main() :
    # texts = ["Hello, world! This is a test."  , "NLP is fascinating... isn't it?" , "Let's see how it handles 123 numbers and punctuation!"]
    #
    # simple_tokenizer = SimpleTokenizer()
    # regex_tokenizer = RegexTokenizer()
    #
    # for text in texts :
    #     # token = simple_tokenizer.tokenize(text)
    #     token = regex_tokenizer.tokenize(text)
    #     print(token)

    dataset_path = f"UD_English-EWT/en_ewt-ud-train.txt"
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[:100]
        lines = [line.strip() for line in lines]
    print(lines)
if __name__ == "__main__":
    main()



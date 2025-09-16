from src.core.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


def main() :
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    tokenize = RegexTokenizer()
    count_vectorizer = CountVectorizer(tokenizer=tokenize)

    vectors = count_vectorizer.fit_transform(corpus)
    print(vectors)
    print(count_vectorizer.vocabulary)
if __name__ == "__main__":
    main()



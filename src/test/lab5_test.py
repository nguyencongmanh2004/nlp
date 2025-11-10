from sklearn.model_selection import train_test_split

from src.core.regex_tokenizer import RegexTokenizer
from src.models.text_classifier import TextClassifier
from src.representations.count_vectorizer import CountVectorizer


def main() :
    texts = [
        # Positive (1)
        "This movie was absolutely amazing, I loved every moment!",
        "The acting was fantastic and the story was heartwarming.",
        "What a brilliant performance, truly inspiring.",
        "A masterpiece of modern cinema, highly recommend!",
        "The soundtrack and visuals were stunning.",
        "Such a great experience, I would watch it again.",
        "Beautiful storytelling and well-developed characters.",
        "This film made me cry in a good way, so touching.",
        "A truly enjoyable movie from start to finish.",
        "One of the best films I've seen this year.",
        "Excellent direction and perfect pacing throughout.",
        "The humor was clever and the dialogue was sharp.",
        "An unforgettable experience, simply wonderful.",
        "Loved the cinematography and emotional depth.",
        "Highly entertaining and beautifully executed.",

        # Negative (0)
        "This movie was terrible, I regret watching it.",
        "The plot made no sense and the characters were flat.",
        "Such a boring experience, I almost fell asleep.",
        "The dialogue was awful and the acting was wooden.",
        "What a complete waste of time and money.",
        "The film was way too long and painfully slow.",
        "I couldn’t connect with the story at all.",
        "Everything felt so forced and unrealistic.",
        "The jokes didn’t land and it wasn’t funny at all.",
        "Visually dull and emotionally empty.",
        "Poor editing ruined the entire movie.",
        "Disappointing from beginning to end.",
        "I was expecting more, but it was a total letdown.",
        "The ending was predictable and unoriginal.",
        "Not worth watching, I’d skip it for sure."
    ]
    labels = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 15 positive
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # 15 negative
    ]

    X_train , X_test , y_train , y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer=tokenizer)
    text_classifier = TextClassifier(vectorizer=vectorizer)
    text_classifier.fit(X_train, y_train)
    pre = text_classifier.predict(X_test)
    eval = text_classifier.evaluate(y_test, pre)
    print(eval)
if __name__ == "__main__" :
    main()




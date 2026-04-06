import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from src.processing.clean_text import clean_text

nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


def get_sentiment(text):

    cleaned = clean_text(text)

    scores = sia.polarity_scores(cleaned)

    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"

    elif compound <= -0.05:
        label = "Negative"

    else:
        label = "Neutral"

    return {
        "cleaned_text": cleaned,
        "sentiment_label": label,
        "sentiment_score": compound,
        "pos_score": scores["pos"],
        "neu_score": scores["neu"],
        "neg_score": scores["neg"],
    }
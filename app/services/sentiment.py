from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        mood = "POSITIVE"
    elif polarity < -0.2:
        mood = "NEGATIVE"
    else:
        mood = "NEUTRAL"

    return {"score": polarity, "mood": mood}

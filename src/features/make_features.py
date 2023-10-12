from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import string


def get_french_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return set(stopwords.words('french'))


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation)).split()


def make_features(df, task, config):
    y = get_output(df, task)
    X = df["video_name"]

    steps = []
    """
    if config.get("Features"):
        for feature in config["Features"]:
            if feature == "No-punctuation":
                steps.append(["remove_punctuation", CountVectorizer(tokenizer=remove_punctuation)])
            elif feature == "Stop-word":
                steps.append(["count_vectorizer", CountVectorizer(stop_words=get_french_stopwords())])
        # If no Features are recognized or no CountVectorizer is added, don't forget to add this at least
        isVectorized = False
        for step in steps:
            if step[0] == "count_vectorizer":
                isVectorized = True
        if not isVectorized:
            steps.append(["count_vectorizer", CountVectorizer()])

    """
    if not config:
        steps.append(["count_vectorizer", CountVectorizer()])
    # Get the features
    if config.get("Features") == "Stop-word":
        steps.append(["count_vectorizer", CountVectorizer(stop_words=get_french_stopwords())])
    elif config.get("Features") == "No-punctuation":
        steps.append(["count_vectorizer", CountVectorizer(tokenizer=remove_punctuation)])
    else:
        steps.append(["count_vectorizer", CountVectorizer()])
    return X, y, steps


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer


def preprocess_text(text):
    stemmer = FrenchStemmer()
    stop_words = set(stopwords.words('french'))

    words = word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


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
    if not config:
        steps.append(["count_vectorizer", CountVectorizer()])
    else:
        # Get the features
        if config.get("Features") == "Stop-word":
            steps.append(["count_vectorizer", CountVectorizer(stop_words=get_french_stopwords())])
        elif config.get("Features") == "No-punctuation":
            steps.append(["count_vectorizer", CountVectorizer(tokenizer=remove_punctuation)])
        elif config.get("Features") == "Stemming":
            X = X.apply(preprocess_text)
            steps.append(["count_vectorizer", CountVectorizer()])
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

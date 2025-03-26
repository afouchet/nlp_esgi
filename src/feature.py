import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm


def preprocess_text(text):
    text = remove_stopwords(text)
    text = stem_words(text)
    return text


def remove_stopwords(text):
    words = text.lower().split()
    stop_words = set(stopwords.words("french"))
    return " ".join([word for word in words if word not in stop_words])


def stem_words(text):
    stemmer = PorterStemmer()
    words = text.lower().split()
    return " ".join([stemmer.stem(word) for word in words])


def make_features(df):
    y = df["is_comic"]
    X = df["video_name"].apply(preprocess_text)

    return X, y

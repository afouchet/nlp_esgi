from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
import re
import ast
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer


def preprocess_text(sentence):
    stemmer = FrenchStemmer()
    stop_words = set(stopwords.words('french'))

    words = word_tokenize(sentence)
    filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def get_french_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return set(stopwords.words('french'))


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation)).split()


def tokenize_and_separate_apostrophe(sentence):
    #tokens = nltk.regexp_tokenize(sentence, pattern=r"\b\w+\b|[-]\b|\b[-]\b", gaps=False)
    #tokens = re.findall(r'\b\w+\b| \s', sentence)
    tokens = re.findall(r"[\w-]+", sentence)

    return tokens


def remove_misstranslate(text):
    return [word for word in text if word != '``' or word != "\'\'" or word != "(" or word != ")"]


def word_plus_tag(X):
    sentence = []
    features = []
    for i, x in enumerate(X):
        for word in x:
            sentence.append(word)
            if word.istitle():
                features.append('1')
            else:
                features.append('0')
    return pd.DataFrame({'Word': sentence, 'Tag': features})


def make_features(df, task, config):
    X, y = get_output(df, task)
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
        elif config.get("Features") == "is_starting_word":
            y = []
            for x in X:
                y.append([1] + [0] * (len(x) - 1))
            steps.append(["count_vectorizer", CountVectorizer()])
        elif config.get("Features") == "is_final_word":
            y = []
            for x in X:
                y.append([0] * (len(x) - 1) + [1])
            steps.append(["count_vectorizer", CountVectorizer()])
        elif config.get("Features") == "is_capitalized":
            X = word_plus_tag(X)
            steps.append(["count_vectorizer", CountVectorizer()])
        else:
            steps.append(["count_vectorizer", CountVectorizer()])
    if task == "is_name":
        y = [item for sublist in y for item in sublist]
        y = pd.DataFrame({'Label': y})
    #TODO: Pour l'erreur de is_name, si on met X["Word"], tout fonctionne. Mais on DOIT mettre les deux colonnes
    return X, y, steps


def get_output(df, task):
    X = df["video_name"]
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
        len_y = []
        len_x = []
        y = [ast.literal_eval(item) for item in y]
        y = pd.Series(y)
        X = X.apply(tokenize_and_separate_apostrophe)
        for numbers in y:
            len_y.append(len(numbers))
        for numbers in X:
            len_x.append(len(numbers))
        # Drop the inconsistency
        for i in range(len(len_x)-1, -1, -1):
            if len_x[i] != len_y[i]:
                y = y.drop(index=i)
                X = X.drop(index=i)
        # Debug len of our features: pd.DataFrame({"y":len_y, "x":len_x}).to_csv("TEST.csv")
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return X, y

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def make_model():
    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        ("random_forest", RandomForestClassifier()),
    ])

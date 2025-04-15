from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def make_model():
    return Pipeline(
        [
            ("count_vectorizer", CountVectorizer(min_df=5)),
            ("random_forest", RandomForestClassifier(n_estimators=100, min_samples_split=10)),
        ]
    )


def make_model_old():
    return MyModel()


class MyModel():
    def __init__(self):
        self._vectorizer = CountVectorizer()
        self._clf = RandomForestClassifier()

    def fit(self, X, y):
        X = self._vectorizer.fit_transform(X)
        self._clf.fit(X, y)

        return self

    def predict(self, X):
        X = self._vectorizer.transform(X)

        return self._clf.predict(X)

def predict_at_word_level(
    words: list[str],
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
) -> list[int]:
    inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True)
    
    logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)

    word_labels = []
    word_ids = inputs.word_ids()
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            word_labels.append(predictions[0][idx].item())
            previous_word_idx = word_idx
    
    return word_labels

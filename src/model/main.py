from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import make_features

def make_model(task):
    if task == "is_name":
        # return RandomForestClassifier(max_depth=10, min_samples_split=10)
        # return LogisticRegression()
        return Pipeline([
            ("text_tokenizer", ColumnTransformer(
                [
                    ("prev_text", CountVectorizer(min_df=2), "prev_text"),
                    ("next_text", CountVectorizer(min_df=2), "next_text")
                ],
                remainder='passthrough',
            )),
            # ("random_forest", RandomForestClassifier(max_depth=5)),
            ("logitic_regression", LogisticRegression()),
        ])
    elif task == "comic_name":
        return ComicNameFinder()

    
    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        ("random_forest", RandomForestClassifier()),
    ])


class ComicNameFinder():
    def __init__(self):
        self._model_name = make_model(task="is_name")
        self._model_comic = make_model(task="is_comic_video")

    def fit(self, df, y):
        self._fit_name_model(df)
        self._fit_comic_model(df)
    
    def predict(self, df):
        name_pred_by_video = self._predict_names(df)
        return self._predict_comic_name(name_pred_by_video)

    def _fit_comic_model(self, df):
        X, y = make_features(df, task="is_comic_video")
        self._model_comic.fit(X, y)
    
    def _fit_name_model(self, df):
        X, y = make_features(df, task="is_name")
        self._model_name.fit(X, y)

    def _predict_names(self, df):
        X, _ = make_features(task="is_name", is_test=True)

    

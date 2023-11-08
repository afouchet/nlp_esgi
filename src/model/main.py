from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features.make_features import make_features, revert_token_pred_in_video_name

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
    elif task == "find_comic_name":
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
        df = df.copy()
        df_with_name_pred = self._predict_names(df)
        return self._predict_comic_name(df_with_name_pred)

    def _fit_comic_model(self, df):
        X, y = make_features(df, task="is_comic_video")
        self._model_comic.fit(X, y)
    
    def _fit_name_model(self, df):
        X, y = make_features(df, task="is_name")
        self._model_name.fit(X, y)

    def _predict_names(self, df):
        X, _ = make_features(df, task="is_name", is_test=True)
        pred = self._model_name.predict(X)

        df_with_pred = revert_token_pred_in_video_name(df, pred)
        df["prediction"] = df_with_pred["prediction"]

        df["pred_names"] = df_with_pred.apply(
            lambda row: extract_names_from_labeled_tokens(row.tokens, row.prediction),
            axis=1,
        )

        return df

    def _predict_comic_name(self, df_with_name_pred):
        X, _ = make_features(df_with_name_pred, task="is_comic_video", is_test=True)
        pred = self._model_comic.predict(X)

        id_not_comic = pred == 0

        # Filling case no name
        df_with_name_pred.loc[id_not_comic, "pred_names"] = None
        df_with_name_pred["pred_names"] = df_with_name_pred["pred_names"].apply(lambda name_list: name_list if name_list else [])

        return df_with_name_pred["pred_names"]
    
    def get_params(self, deep=True):
        return {}

def extract_names_from_labeled_tokens(tokens, labels):
    """
    Given tokens & labels (as name or not), extract list of names.

    For example, for tokens
    ["Thomas", "Rivière", "vous", "raconte", "Pierrot", "le", "Fou"]
    and labels
    [1, 1, 0, 0, 1, 0, 1]
    It would consider "Thomas Rivière" to be one name (as tokens are
    successive) and "Pierrot" and "Fou" to be 2 other names.

    It would output ["Thomas Rivière", "Pierrot", "Fou"]
    """
    curr_name = ""
    names = []

    for token, label in zip(tokens, labels):
        if label == 1:
            if curr_name:
                curr_name += " " + token
            else:
                curr_name = token
        else:
            if curr_name:
                names.append(curr_name)
                curr_name = ""

    if curr_name:
        names.append(curr_name)

    return names
    

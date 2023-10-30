from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
import re
import ast
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from nltk import pos_tag, ne_chunk

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
    return list(stopwords.words('french'))


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation)).split()


def tokenize_and_separate_apostrophe(sentence):
    return re.findall(r"[\w-]+", sentence)


def remove_misstranslate(text):
    return [word for word in text if word != '``' or word != "\'\'" or word != "(" or word != ")"]


def word_plus_tag_capitalized(X):
    sentence = []
    features = []
    for x in X:
        for word in x:
            sentence.append(word)
            if word.istitle():
                features.append(1)
            else:
                features.append(0)
    return pd.DataFrame({'Word': sentence, 'Tag': features})


def word_plus_tag_start(X):
    sentence = []
    features = []
    for x in X:
        for j, word in enumerate(x):
            sentence.append(word)
            if j == 0:
                features.append(1)
            else:
                features.append(0)
    return pd.DataFrame({'Word': sentence, 'Tag': features})


def word_plus_tag_end(X):
    sentence = []
    features = []
    for x in X:
        for j, word in enumerate(x):
            sentence.append(word)
            if j == (len(x)-1):
                features.append(1)
            else:
                features.append(0)
    return pd.DataFrame({'Word': sentence, 'Tag': features})


def personnal_parsing(X, y):
    y = [ast.literal_eval(item) for item in y]
    y = pd.Series(y)
    X = X.apply(tokenize_and_separate_apostrophe)
    len_y = [len(numbers) for numbers in y]
    len_x = [len(numbers) for numbers in X]
    # Drop the inconsistency
    for i in range(len(len_x) - 1, -1, -1):
        if len_x[i] != len_y[i]:
            y = y.drop(index=i)
            X = X.drop(index=i)
    # Debug len of our features: pd.DataFrame({"y":len_y, "x":len_x}).to_csv("TEST.csv")
    return X, y


def setup_dataframe(first_df, second_df, third_df):
    # Retrieve the comic name
    comic_name = []
    count_for_third_df = 0
    for i, sentence in enumerate(first_df["Sentence"]):
        temp_name = []
        for _ in sentence:
            if third_df["is_comic"].iloc[i] == 1 and second_df["is_name"].iloc[count_for_third_df] == 1:
                temp_name.append(second_df["X_name"].iloc[count_for_third_df])
            count_for_third_df += 1
        comic_name.append(" ".join(temp_name))
    return comic_name


def extract_names(text):
    tokens = word_tokenize(text, language='french')
    # French not supported :/
    tagged = pos_tag(tokens, lang='eng')
    named_entities_list = ne_chunk(tagged, binary=False)
    names = []
    for entity in named_entities_list:
        if isinstance(entity, nltk.Tree):
            name = " ".join([word for word, tag in entity.leaves()])
            names.append(name)
    return names


def make_features(df, task, config):
    X, y = get_output(df, task)
    steps = []
    if not config:
        steps.append(["count_vectorizer", CountVectorizer()])
    elif config.get("Features") == "Stop-word":
        steps.append(["count_vectorizer", CountVectorizer(stop_words=get_french_stopwords())])
    elif config.get("Features") == "No-punctuation":
        steps.append(["count_vectorizer", CountVectorizer(tokenizer=remove_punctuation)])
    elif config.get("Features") == "Stemming":
        X = X.apply(preprocess_text)
        steps.append(["count_vectorizer", CountVectorizer()])
    elif config.get("Features") == "is_starting_word":
        X = word_plus_tag_start(X)
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(), 'Word'),
                ('num', StandardScaler(), ['Tag'])
            ],
            remainder='passthrough'
        )
        steps.append(["count_vectorizer", preprocessor])
    elif config.get("Features") == "is_final_word":
        X = word_plus_tag_end(X)
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(), 'Word'),
                ('num', StandardScaler(), ['Tag'])
            ],
            remainder='passthrough'
        )
        steps.append(["count_vectorizer", preprocessor])
    elif config.get("Features") == "is_capitalized":
        X = word_plus_tag_capitalized(X)
        # /!\ Beware, you need to be precise if there is a column with caracters. We need to make a transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(), 'Word'),
                ('num', StandardScaler(), ['Tag'])
            ],
            remainder='passthrough'
        )
        steps.append(["count_vectorizer", preprocessor])
    elif config.get("Features") == "mix_model":
        return extract_comic_and_person_names(df)
    elif config.get("Features") == "named_entity":
        X = pd.DataFrame({"video_name": df['video_name'].apply(extract_names)})
        X = pd.DataFrame({"video_name": X['video_name'].apply(lambda x: ' '.join(x))})
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(), 'video_name')
            ],
            remainder='passthrough'
        )
        steps.append(["count_vectorizer", preprocessor])
        y = df['is_comic']
    else:
        steps.append(["count_vectorizer", CountVectorizer()])
    if task == "is_name":
        y = [item for sublist in y for item in sublist]
        y = pd.DataFrame({'Label': y})
        y = y.values.ravel()
    return X, y, steps


def extract_comic_and_person_names(df):
    # Set the is_comic feature
    y_comic = df["is_comic"]
    pipeline = Pipeline([("count_vectorizer", CountVectorizer(stop_words=get_french_stopwords())),
                         (["loaded_model", GradientBoostingClassifier()])])
    pipeline.fit(df["video_name"], y_comic)
    prediction_comic = pipeline.predict(df["video_name"])

    # Set the is_name feature
    y_name = df["is_name"]
    y_name = [ast.literal_eval(item) for item in y_name]
    y_name = [item for list_item in y_name for item in list_item]
    y_name = pd.DataFrame({'Label': y_name})
    y_name = y_name.values.ravel()
    X_name = df["video_name_parsed"]
    X_name = [ast.literal_eval(item) for item in X_name]
    X_name = [(item, 1) if item.istitle() else (item, 0) for list_item in X_name for item in list_item]
    X_name = pd.DataFrame(X_name, columns=['Word', 'Tag'])
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'Word')
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline([("count_vectorizer", preprocessor),
                         (["loaded_model", GradientBoostingClassifier()])])
    pipeline.fit(X_name, y_name)
    prediction_name = pipeline.predict(X_name)
    X_predicted_name = pd.DataFrame({"X_name": X_name["Word"], "is_name": prediction_name})

    # Retrieve the result and send it
    sentence_df = pd.DataFrame({"Sentence": [ast.literal_eval(item) for item in df["video_name_parsed"]]})
    predicted_name_comic = pd.DataFrame({"comic_name": (setup_dataframe(sentence_df, X_predicted_name, pd.DataFrame({"is_comic":prediction_comic})))})
    return predicted_name_comic, None, None


def get_output(df, task):
    X = df["video_name"]
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
        X, y = personnal_parsing(X, y)
    elif task == "find_comic_name":
        y = df["comic_name"]
        y = [ast.literal_eval(item) for item in y]
        y = pd.Series(y)
        X = X.apply(tokenize_and_separate_apostrophe)
    else:
        raise ValueError("Unknown task")

    return X, y

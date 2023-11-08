import json


def make_features(df, task, is_test=False):
    if task == "is_comic_video":
        return make_features__is_comic(df, is_test)
    elif task == "is_name":
        return make_features__is_name(df, is_test)
    elif task == "find_comic_name":
        return make_features__comic_name(df, is_test)

    return X, y


def make_features__is_comic(df, is_test):
    X = df["video_name"]
    if is_test:
        y = None
    else:
        y = df["is_comic"]

    return X, y


def make_features__is_name(df, is_test):
    df = df.copy()
    if is_test:
        y = None
    else:
        df["is_name"] = df["is_name"].apply(json.loads)
        y = df.explode("is_name")["is_name"].astype("int")
        
    df["tokens"] = df["tokens"].apply(json.loads)

    df["surrounding_text"] = df.tokens.apply(
        lambda toks: describe_prev_and_next_text(toks, 3)
    )

    df = explode_by_token(df)

    df["is_capitalized"] = df.tokens.apply(lambda txt: 1 * txt[:1].isupper())

    X = df[
        ["prev_text", "next_text", "is_capitalized", "token_place_till_end", "token_place"]
    ]

    return X, y


def make_features__comic_name(df, is_test):
    df = df.copy()
    y = df.pop("comic_name").apply(json.loads)
    return df, y


def describe_prev_and_next_text(tokens, nb_words):
    desc = []
    nb_tokens = len(tokens)
    for i, tok in enumerate(tokens):
        prev_words = " ".join(tokens[max(i - nb_words, 0):i])
        next_words = " ".join(tokens[i+1: i+1+nb_words])

        desc.append({
            "prev_words": prev_words,
            "next_words": next_words,
            "token_place": i,
            "token_place_till_end": nb_tokens - i,
        })

    return desc


def explode_by_token(df):
    df = df.explode(["tokens", "surrounding_text"])
    df["prev_text"] = df["surrounding_text"].apply(lambda desc: desc["prev_words"])
    df["next_text"] = df["surrounding_text"].apply(lambda desc: desc["next_words"])
    df["token_place"] = df["surrounding_text"].apply(lambda desc: desc["token_place"])
    df["token_place_till_end"] = df["surrounding_text"].apply(lambda desc: desc["token_place_till_end"])
    return df
    

def revert_token_pred_in_video_name(df, pred_by_token):
    df = df.copy()
    df["tokens"] = df["tokens"].apply(json.loads)
    df_exploded = df.explode("tokens")
    df_exploded["prediction"] = pred_by_token

    df_with_group = (
        df_exploded.groupby("video_name", as_index=False)["prediction"]
        .apply(list)
    )

    df_with_group["tokens"] = df["tokens"]

    return df_with_group
    

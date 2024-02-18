import click
import json
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.make_model import make_model
from model.load_model import load_model
from sklearn.pipeline import Pipeline

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train_v2.csv", help="File training data")
@click.option("--model_dump_filename", default="model/saved/dump.joblib", help="File to dump model")
@click.option("--config", default={}, help="Config to use")
def train(task, input_filename, model_dump_filename, config):
    try:
        config = json.loads(config)
    except Exception:
        print("WARNING: Argument config not well parsed.")
        config = None
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y, steps = make_features(df, task, config)

    # Object with .fit, .predict methods
    model = make_model(config, steps)
    model.fit(X, y)

    return joblib.dump(model, model_dump_filename)
    #return model.named_steps["loaded_model"].dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train_v2.csv", help="File training data")
@click.option("--model_dump_filename", default="model/saved/dump.joblib", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
@click.option("--config", default={}, help="Config to use")
def test(task, input_filename, model_dump_filename, output_filename, config):
    try:
        config = json.loads(config)
    except Exception:
        print("WARNING: Argument config not well parsed.")
        config = None
    df = make_dataset(input_filename)
    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y, _ = make_features(df, task, config)
    # Object with .fit, .predict methods
    pipeline = load_model(model_dump_filename)
    #X = pipeline.named_steps["count_vectorizer"].transform(X.values)
    pipeline.fit(X, y)
    scores = pipeline.predict(X)
    np.savetxt(output_filename, (scores), delimiter=',')
    return evaluate_model(pipeline, X, scores)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train_v2.csv", help="File training data")
@click.option("--config", default={}, help="Config to use")
def evaluate(task, input_filename, config):
    try:
        config = json.loads(config)
    except Exception:
        print("WARNING: Argument config not well parsed.")
        config = None
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y, steps = make_features(df, task, config)

    if config.get("Features") == "mix_model":
        evaluate_part3(X["comic_name"], pd.DataFrame({"comic_name": [ast.literal_eval(item) for item in df["comic_name"]]})["comic_name"])
    else:
        # Object with .fit, .predict methods
        model = make_model(config, steps)

        # Run k-fold cross validation. Print results
        return evaluate_model(model, X, y)



def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores



def evaluate_part3(X, y):
    max_score = len(X)
    score = 0
    for i, _ in enumerate(X):
        if len(y.values[i]) == 0 and X.values[i] == "":
            score += 1
        elif len(y.values[i]) == 0:
            pass
        elif X.values[i] == y.values[i][0]:
            score += 1
    print(f"Got accuracy {(score/max_score)*100}%")

def train_model(model, X, y):
    # Scikit learn has function for train model
    model = model.fit(X, y)

    return model


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

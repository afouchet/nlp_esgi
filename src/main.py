import click
import json
import numpy as np
from sklearn.model_selection import cross_val_score

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.make_model import make_model
from model.load_model import load_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="model/dump.json", help="File to dump model")
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
    model = load_model(steps, model_dump_filename)
    model.fit(X, y)

    return model.named_steps["loaded_model"].dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="model/dump.json", help="File to dump model")
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
    X, y, steps = make_features(df, task, config)

    # Object with .fit, .predict methods
    model = load_model(steps, model_dump_filename)

    return model.named_steps["loaded_model"].dump(output_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
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

    # Object with .fit, .predict methods
    model = make_model(config, steps)

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


def train_model(model, X, y):
    # Scikit learn has function for train model
    model = model.fit(X, y)

    return model


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

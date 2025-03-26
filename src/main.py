import click
import joblib
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_validate

from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model()
    model.fit(X, y)

    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    model = joblib.load(model_dump_filename)

    df = make_dataset(input_filename)
    X, y = make_features(df)

    predictions = model.predict(X)

    print("Got accuracy", (y == predictions).mean() * 100)
    print("Got log loss", log_loss(y, predictions))


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Run k-fold cross validation. Print results
    scoring = ['neg_log_loss', 'accuracy']
    cv_results = cross_validate(
        model, X, y, scoring=scoring, cv=5, return_train_score=True,
        verbose=1, n_jobs=-1,
    )

    print(
        f"Log loss {np.mean(cv_results['test_neg_log_loss']):.5f} +/- {np.std(cv_results['test_neg_log_loss']):.5f}"
    )
    print(
        f"accuracy {np.mean(cv_results['test_accuracy']):.5f} +/- {np.std(cv_results['test_accuracy']):.5f}"
    )


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

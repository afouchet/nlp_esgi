import click
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from src.data.make_dataset import make_dataset
from src.features.make_features import make_features, revert_token_pred_in_video_name
from src.model.main import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model(task)
    model.fit(X, y)

    return model.dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model(task)

    import ipdb
    ipdb.set_trace()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y, task)


def evaluate_model(model, X, y, task):
    if task == "is_comic_video":
        scoring = "accuracy"
    elif task == "is_name":
        scoring = "neg_log_loss"
    elif task == "find_comic_name":
        scoring = make_scorer(compare_comic_list)

    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring=scoring, n_jobs=1)

    print(f"Got accuracy {np.mean(scores)}%")

    return scores


def compare_comic_list(y_true, y_pred):
    score = 0
    for true_list, pred_list in zip(y_true, y_pred):
        if not pred_list:
            score += not true_list
        else:
            score += all(
                any(name in true_name for true_name in true_list)
                for name in pred_list
            )
            
    return score / len(y_true)


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

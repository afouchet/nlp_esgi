from sklearn.pipeline import Pipeline

from .bayes import Bayes
from .linear import Linear
from .randomforest import RandomForest
from .XGBoost import XGBoost


def make_model(config=None, steps=[]):
    if not config:
        steps.append(["loaded_model", RandomForest()])
    else:
        # Get the model
        if config.get("Model") == "Bayes":
            steps.append(["loaded_model", Bayes()])
        elif config.get("Model") == "Linear":
            steps.append(["loaded_model", Linear()])
        elif config.get("Model") == "XGBoost":
            steps.append(["loaded_model", XGBoost()])
        elif config.get("Model") == "Random_Forest":
            steps.append(["loaded_model", RandomForest()])
        else:
            steps.append(["loaded_model", RandomForest()])
    return Pipeline(steps)

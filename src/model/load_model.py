import os.path
import joblib

from .dumb_model import DumbModel


def load_model(model_filename="model/saved/dump.joblib"):
    if os.path.exists(model_filename):
        return joblib.load(open(model_filename, "rb"))
    else:
        return DumbModel()


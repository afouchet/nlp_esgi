import os.path
import pickle

from .dumb_model import DumbModel


def load_model(model_filename="model/dump.json"):
    if os.path.exists(model_filename):
        return pickle.load(open(model_filename, "rb"))
    else:
        return DumbModel()


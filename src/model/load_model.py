import os.path

from .dumb_model import DumbModel
from sklearn.pipeline import Pipeline


def load_model(steps=[], model_filename="model/dump.json"):
    if os.path.exists(model_filename):
        # Comment lire le json et le traduire en modèle?
        # Autrement, si nous n'attendons pas nécessairement des modèles en JSON:
        # steps.append(["loaded_model", ])
        steps.append(["loaded_model", DumbModel()])
    else:
        steps.append(["loaded_model", DumbModel()])

    return Pipeline(steps)

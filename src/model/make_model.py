from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def make_model(config=None, steps=[]):
    if not config:
        steps.append(["random_forest", RandomForestClassifier()])
    else:
        # Get the model
        if config.get("Model") == "Bayes":
            steps.append(["naive_bayes", MultinomialNB()])
        elif config.get("Model") == "Linear":
            steps.append(["linear", LogisticRegression()])
        elif config.get("Model") == "XGBoost":
            steps.append(["XGBoost", GradientBoostingClassifier()])
        elif config.get("Model") == "Random_Forest":
            steps.append(["random_forest", RandomForestClassifier()])
        else:
            steps.append(["random_forest", RandomForestClassifier()])
    return Pipeline(steps)

from sklearn.ensemble import GradientBoostingClassifier
import pickle


class XGBoost():
    classifier = GradientBoostingClassifier()

    def load(self, model):
        self.classifier = model

    def fit(self, x_train_tfidf, y_train):
        self.classifier.fit(x_train_tfidf, y_train)

    def predict(self, x_test_tfidf):
        return self.classifier.predict(x_test_tfidf)

    def dump(self, filename_output):
        pickle.dump(self.classifier, open(filename_output, "wb"))

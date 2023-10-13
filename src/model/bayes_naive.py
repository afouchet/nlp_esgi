from sklearn.naive_bayes import MultinomialNB


class Bayes:
    classifier = MultinomialNB()
    def fit(self, x_train_tfidf, y_train):
        self.classifier.fit(x_train_tfidf, y_train)

    def predict(self, x_test_tfidf):
        y_pred = self.classifier.predict(x_test_tfidf)
        return y_pred

    def dump(self, filename_output):
        pass

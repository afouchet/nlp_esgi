import json


class DumbModel:
    """Dumb model always predict 0"""
    def fit(self, X, y):
        pass

    def predict(self, X):
        return [0] * len(X)

    def dump(self, filename_output):
        # Serializing json
        json_object = json.dumps({}, indent=4)

        # Writing to sample.json
        with open(filename_output, "w") as outfile:
            outfile.write(json_object)

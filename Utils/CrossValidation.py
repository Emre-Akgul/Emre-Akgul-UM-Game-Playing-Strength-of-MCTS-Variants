from Utils.Utils import root_mean_squared_error
import pandas as pd

class KFoldCrossValidation:
    def __init__(self, model, X, y, k=5):
        self.model = model
        self.X = X
        self.y = y
        self.k = k
        self.splits = self._split()
        self.scores = []
        self._kFold()

    def _split(self):
        n = len(self.X)
        splits = []
        for i in range(self.k):
            start = i * n // self.k
            end = (i + 1) * n // self.k
            X_train = pd.concat([self.X.iloc[:start], self.X.iloc[end:]])
            y_train = pd.concat([self.y.iloc[:start], self.y.iloc[end:]])
            X_test = self.X.iloc[start:end]
            y_test = self.y.iloc[start:end]
            splits.append((X_train, y_train, X_test, y_test))
        return splits


    def _kFold(self):
        for X_train, y_train, X_test, y_test in self.splits:
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            self.scores.append(root_mean_squared_error(y_test, predictions))

    def scores(self):
        return self.scores
    
    def mean_score(self):
        return sum(self.scores) / len(self.scores)

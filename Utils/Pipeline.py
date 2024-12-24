from Models.Model import Model
from Utils.Preprocessor import Preprocessor

class Pipeline(Model):
    def __init__(self, preprocessor, model):
        """
        Initialize the Model.
        I hate python interfaces.
        """
        self.preprocessor = preprocessor
        self.model = model

    def fit(self, X, y):
        """
        Method for fitting the model to data.
        Must be implemented by all subclasses.
        """
        X_p = self.preprocessor.fit_transform(X)
        self.model.fit(X_p, y)

    def predict(self, X):
        """
        Method for predicting values.
        Must be implemented by all subclasses.
        """
        return self.model.predict(self.preprocessor.transform(X))

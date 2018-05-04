from sklearn.base import BaseEstimator

class SklearnClassifier():
    """description of class"""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
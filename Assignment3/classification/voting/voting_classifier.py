from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from classification.classifier import Classifier
from classification.sklearn_classifier import SklearnClassifier
from sklearn.base import BaseEstimator


class Voting(Classifier, SklearnClassifier):
    """description of class"""

    def __init__(self, dataset, estimators: [(str, BaseEstimator)], logger=None, voting='hard', weights=None, *args,
                 **kwargs):
        """
        Parameters
        ----------
        voting: str, {‘hard’, ‘soft’} (default=’hard’) 
            If ‘hard’, uses predicted class labels for majority rule voting. Else if ‘soft’, 
            predicts the class label based on the argmax of the sums of the predicted probabilities, 
            which is recommended for an ensemble of well-calibrated classifiers.
        weights: array-like, shape = [n_classifiers], optional (default=`None`)
            Sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting) 
            or class probabilities before averaging (soft voting). Uses uniform weights if None.
        """

        self.classifier = VotingClassifier(estimators, voting=voting, weights=weights)

        SklearnClassifier.__init__(self, self.classifier)
        Classifier.__init__(self, dataset, logger)

    def fit(self):
        self.classifier.fit(self.ds.x_train, self.ds.y_train)

    def update(self, x, y):
        self.classifier.fit(x, y)
        self.save_online_model('voting')

    def predict(self, x):
        return self.classifier.predict(x)

    def predict_proba(self, x):
        return self.classifier.predict_proba(x)

    def validate(self):
        accuracy = self.classifier.score(self.ds.x_val, self.ds.y_val)
        self.logger.log_and_print(f"accuracy: \t {accuracy:04.2f}")
        return accuracy

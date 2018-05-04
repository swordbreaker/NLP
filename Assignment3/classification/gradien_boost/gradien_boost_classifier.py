import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from classification.classifier import Classifier
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from classification.util.logger import Logger
from classification.sklearn_classifier import SklearnClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


class GradienBoost(Classifier, SklearnClassifier):
    """Class for gradient boosting"""

    def __init__(self, dataset: "DataSet", n_estimators=120, verbose=0, model=None, logger: "Logger" = None):
        self.scores = scores = ['recall_weighted', 'precision_weighted']
        self.tuned_parameters = {'loss': ['deviance'],
                                 'learning_rate': [0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
                                 'n_estimators': [10, 30, 50, 100, 150, 200],
                                 'max_depth': [2, 3, 4, 5, 6, 7, None]}
        if model == None:
            self.classifier = GradientBoostingClassifier(n_estimators=100, max_depth=5, loss='deviance',
                                                         learning_rate=0.1)
            # self.classifier = GradientBoostingClassifier(max_depth=2, n_estimators=n_estimators, verbose=verbose)
        else:
            self.classifier = model

        SklearnClassifier.__init__(self, self.classifier)
        Classifier.__init__(self, dataset, logger=logger)

    def find_best_estimaotrs(self):
        self.classifier.fit(self.ds.x_train, self.ds.y_train)
        errors = [mean_squared_error(self.ds.y_val, y_pred) for y_pred in self.classifier.staged_predict(self.ds.x_val)]
        best_n_estimators = np.argmin(errors)
        return best_n_estimators

    def fit(self):
        self.classifier.fit(self.ds.x_train, self.ds.y_train)

    def update(self, x, y):
        self.classifier.fit(x, y)
        self.save_online_model('gradient_boost')

    def hyper_parameter_tuning(self):
        for score in self.scores:
            self.logger.log_and_print("# Tuning hyper-parameters for %s" % score)
            self.logger.log_and_print()
            x_train, y_train = self.ds.cross_validation()
            clf = RandomizedSearchCV(GradientBoostingClassifier(), self.tuned_parameters,
                                     scoring=score, n_iter=80, cv=10)
            clf.fit(x_train, y_train)
            self.logger.log_and_print("Best parameters set found on development set:")
            self.logger.log_and_print()
            self.logger.log_and_print(clf.best_params_)
            self.logger.log_and_print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            self.logger.log_and_print()

            self.logger.log_and_print("Detailed classification report:")
            self.logger.log_and_print()
            self.logger.log_and_print("The model is trained on the full development set.")
            self.logger.log_and_print("The scores are computed on the full evaluation set.")
            self.logger.log_and_print()
            y_true, y_pred = self.ds.y_test, clf.predict(self.ds.x_test)
            self.logger.log_and_print(set(y_true) - set(y_pred))
            self.logger.log_and_print(classification_report(y_true, y_pred))
            self.logger.log_and_print()
            self.classifier = clf.best_estimator_
            self.estimator = self.classifier

    def validate(self):
        accuracy = self.classifier.score(self.ds.x_test, self.ds.y_test)
        self.logger.log_and_print(f"accuracy: \t {accuracy:04.2f}")
        return accuracy

    def predict(self, x: any) -> [any]:
        return self.classifier.predict(x)

    def predict_proba(self, x):
        return self.classifier.predict_proba(x)

    def save(self, path: str):
        joblib.dump(self.classifier, path)

    @staticmethod
    def load(path: str, dataset: "DataSet") -> "GradienBoost":
        model = joblib.load(path)
        return GradienBoost(dataset, model=model)

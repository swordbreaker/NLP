import numpy as np
from classification.classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from classification.util.logger import Logger
from classification.sklearn_classifier import SklearnClassifier


class SupportingVectorMachine(Classifier, SklearnClassifier):
    """Class for supporting vector machine"""

    def __init__(self, dataset: "DataSet", loss='squared_hinge', max_iter=1000, verbose=0, model=None,
                 logger: "Logger" = None):
        # self.scores = scores = ['recall_weighted', 'precision_weighted']
        self.scores = scores = ['precision_weighted']
        self.tuned_parameters = {'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4, 'auto'],
                                 'kernel': ['rbf', 'linear', 'poly'], 'class_weight': ['balanced', None],
                                 'degree': [3, 4, 5]}
        if model == None:
            self.classifier = SVC()
        else:
            self.classifier = model

        SklearnClassifier.__init__(self, self.classifier)
        Classifier.__init__(self, dataset, logger=logger)
        # super(Classifier, self).__init__()
        # super(SklearnClassifier, self).__init__(self.classifier)
        # super(Classifier, self).__init__(dataset, logger)

    def fit(self):
        for score in self.scores:
            clf = SVC(kernel='linear', gamma=0.001, degree=3, class_weight=None, C=10, probability=True)
            clf.fit(self.ds.x_train, self.ds.y_train)
            self.classifier = clf
            self.estimator = self.classifier

    def update(self, x, y):
        self.classifier.fit(x, y)
        self.save_online_model('svm')

    def hyper_parameter_tuning(self):
        for score in self.scores:
            self.logger.log_and_print("# Tuning hyper-parameters for %s" % score)
            self.logger.log_and_print()
            x_train, y_train = self.ds.cross_validation()

            clf = RandomizedSearchCV(SVC(), self.tuned_parameters,
                                     scoring=score, n_iter=200, cv=10)
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
    def load(path: str, dataset: "DataSet" = None, logger: "Logger" = None) -> "SupportingVectorMachine":
        model = joblib.load(path)
        return SupportingVectorMachine(dataset, model=model, logger=logger)

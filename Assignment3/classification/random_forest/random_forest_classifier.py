import numpy as np
from classification.classifier import Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from classification.sklearn_classifier import SklearnClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


class RandomForest(Classifier, SklearnClassifier):
    """class for random forest classification"""

    def __init__(self, dataset: "DataSet", n_estimators=500, max_leaf_nodes=16, verbose=0,
                 model: RandomForestClassifier = None, logger=None):
        self.scores = scores = ['recall_weighted', 'precision_micro', 'precision_weighted']
        self.tuned_parameters = {'criterion': ['gini', 'entropy'],
                                 'class_weight': ['balanced', None],
                                 'n_estimators': [10, 30, 50, 100, 150, 200],
                                 'max_depth': [2, 3, 4, 5, 6, 7, None],
                                 'bootstrap': [True, False]}
        if model == None:
            self.classifier = RandomForestClassifier(n_estimators=150, max_depth=5, criterion='gini', class_weight=None,
                                                     bootstrap=True)
            # self.classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=16, n_jobs=-1,
            #                                         verbose=verbose)
        else:
            self.classifier = model

        SklearnClassifier.__init__(self, self.classifier)
        Classifier.__init__(self, dataset, logger=logger)

    def fit(self):
        self.classifier.fit(self.ds.x_train, self.ds.y_train)

    def update(self, x, y):
        self.classifier.fit(x, y)
        self.save_online_model('random_forest')

    def hyper_parameter_tuning(self):
        for score in self.scores:
            self.logger.log_and_print("# Tuning hyper-parameters for %s" % score)
            self.logger.log_and_print()
            x_train, y_train = self.ds.cross_validation()

            clf = RandomizedSearchCV(RandomForestClassifier(), self.tuned_parameters,
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
    def load(path: str, dataset: "DataSet") -> "RandomForest":
        model = joblib.load(path)
        return RandomForest(dataset, model=model)

import numpy as np
from classification.classifier import Classifier
from sklearn.externals import joblib
from classification.sklearn_classifier import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report


class MultinomialNaiveBayes(Classifier, SklearnClassifier):
    """description of class"""

    def __init__(self, dataset: "DataSet", verbose=0, model: MultinomialNB = None, logger=None):
        self.scores = scores = ['precision_weighted']
        self.tuned_parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
                                 'tfidf__use_idf': [True, False],
                                 'tfidf__smooth_idf': [True, False],
                                 'tfidf__sublinear_tf': [True, False],
                                 'clf__alpha': [1e-2, 1e-3, 1e-4],
                                 'clf__fit_prior': [True, False]}
        if model == None:
            self.classifier = MultinomialNB()
        else:
            self.classifier = model

        SklearnClassifier.__init__(self, self.classifier)
        Classifier.__init__(self, dataset, logger=logger)

    def fit(self):
        self.classifier.fit(self.ds.x_train, self.ds.y_train)

    def update(self, x, y):
        self.classifier.partial_fit(x, y)
        self.save_online_model('multinomial_naive_bayes')

    def hyperparameter(self):
        self.ds.x_test = self.ds.x_test_text
        self.ds.x_train = self.ds.x_train_text
        self.ds.x_val = self.ds.x_val_text

        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())
                             ])
        for score in self.scores:
            clf = RandomizedSearchCV(text_clf, self.tuned_parameters, scoring=score, n_iter=192)
            x_train, y_train = self.ds.cross_validation()
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

    def update(self, x, y):
        self.classifier.partial_fit(x, y)

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
    def load(path: str, dataset: "DataSet") -> "MultinomialNaiveBayes":
        model = joblib.load(path)
        return MultinomialNaiveBayes(dataset, model=model)

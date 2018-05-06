import numpy as np
from classification.data_set import DataSet
from sklearn.metrics import precision_recall_fscore_support
from classification.util import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from classification.util.logger import Logger
from itertools import compress
import time


class Classifier():
    """abstract class for a classifier"""

    def __init__(self, dataset: "DataSet", logger: "Logger" = None, *args, **kwargs):
        self.ds = dataset
        self.logger = logger

    def fit(self):
        raise NotImplementedError("Should have implemented this")

    def update(self, x, y):
        raise NotImplementedError("Should have implemented this")

    def validate(self):
        raise NotImplementedError("Should have implemented this")

    def predict(self, x):
        raise NotImplementedError("Should have implemented this")

    def predict_proba(self, x):
        raise NotImplementedError("Should have implemented this")

    def save(self, path: str):
        raise NotImplementedError("Should have implemented this")

    def metrics(self):
        y_true = self.ds.y_test
        y_pred = self.predict(self.ds.x_test)

        (precision, recall, fscore, _) = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        if self.logger is None:
            print(f"precision: \t {precision:04.2f}")
            print(f"recall: \t {recall:04.2f}")
            print(f"fscore: \t {fscore:04.2f}")
        else:
            self.logger.log_and_print(f"precision: \t {precision:04.2f}")
            self.logger.log_and_print(f"recall: \t {recall:04.2f}")
            self.logger.log_and_print(f"fscore: \t {fscore:04.2f}")

        return (precision, recall, fscore)

    #def print_wrong_test(self):
    #    idx_offset_length = len(self.ds.y_train) + len(self.ds.y_val)
    #    idx_offset = np.zeros(idx_offset_length, dtype=bool)
    #    y_true = self.ds.y_test
    #    y_pred = self.predict(self.ds.x_test)
    #    correct_predicted = np.equal(y_true, y_pred)
    #    false_predicted = np.invert(correct_predicted)
    #    filter_indexes = np.append(idx_offset, false_predicted)
    #    wrong_predicted = list(compress(self.ds.raw_data, filter_indexes))
    #    for prediction in wrong_predicted:
    #        text = u''.join(prediction).encode('utf-8')
    #        self.logger.log_and_print(text)

    def plot_confusion_matrix(self, show_plot=True):
        y_true = self.ds.y_test
        y_pred = self.predict(self.ds.x_test)

        class_names = []
        if self.ds.class_names == None:
            le = LabelEncoder()
            le.fit(y_true)
            class_names = le.classes_
        else:
            class_names = self.ds.class_names

        if self.logger is None:
            path = None
        else:
            path = self.logger.get_log_path("confusion_matrix", ".png")

        confusion_matrix.create_and_plot_confusion_matrix(y_true, y_pred, class_names, save_path=path,
                                                          show_plot=show_plot)

    def save_online_model(self, model: str):
        str_time = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        path = f'../../data/online_learning/{model}/{str_time}'
        self.save(f'{path}_{model}_model.pkl')
        file = open(f'{path}_metrics', 'w+')
        print(f'Accuracy: {self.classifier.score(self.ds.x_test, self.ds.y_test)}', file=file)
        (precision, recall, fscore) = self.metrics()
        print(f'Precision: {precision}', file=file)
        print(f'Precision: {recall}', file=file)
        print(f'Precision: {fscore}', file=file)
        file.close()

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

class DataSet(object):
    """class for out datasets"""

    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_val: np.ndarray,
                 y_val: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray,
                 class_names: [str] = None,
                 raw_data: [str] = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.class_names = class_names
        self.raw_data = raw_data
        self.x_train_text = []
        self.x_val_text = []
        self.x_test_text = []

    def shuffle(self):
        np.random.seed(42)
        idx = np.arange(self.x_train.shape[0])
        np.random.shuffle(idx)

        self.x_train = self.x_train[idx]
        self.y_train = self.y_train[idx]
        self.x_val = self.x_val[idx]
        self.y_val = self.y_val[idx]
        self.x_test = self.x_test[idx]
        self.y_test = self.y_test[idx]

    def cross_validation(self):
        x = np.concatenate((self.x_train, self.x_val), axis=0)
        y = np.concatenate((self.y_train, self.y_val), axis=0)
        return x, y

    def add_text_data(self, texts: [str]):
        self.x_train_text = texts[:self.x_train.shape[0]]
        self.x_val_text = texts[self.x_train.shape[0]:self.x_train.shape[0] + self.x_val.shape[0]]
        self.x_test_text = texts[self.x_train.shape[0] + self.x_val.shape[0]:]

    @staticmethod
    def fromtf(dataset):
        """To import a dataset from tensorflow"""
        x_train = dataset.train.images
        y_train = dataset.train.labels
        x_val = dataset.validation.images
        y_val = dataset.validation.labels
        x_test = dataset.test.images
        y_test = dataset.test.labels
        return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)

    @staticmethod
    def from_np_array(x: np.ndarray, y: np.ndarray, class_names: [str] = None, raw_data: [str] = None, p_train=0.6,
                      p_val=0.2, shuffle=False):

        if shuffle:
            np.random.seed(42)
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)
            x = x[idx]
            y = y[idx]

        n = x.shape[0]
        n_train = int(n * p_train)
        n_val = int(n * p_val)

        x_train = x[:n_train]
        y_train = y[:n_train]
        x_val = x[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        x_test = x[n_train + n_val:]
        y_test = y[n_train + n_val:]

        return DataSet(x_train, y_train, x_val, y_val, x_test, y_test, class_names, raw_data)

    def plot_distribution(self, set: str):
        """
            set: 'train', 'val', 'test' or 'all'
        """
        classes = self.class_names
        labels = None
        if set == 'train':
            labels = self.y_train
        elif set == 'val':
            labels = self.y_val
        elif set == 'test':
            labels = self.y_test
        elif set == 'all':
            labels = np.concatenate((self.y_train, self.y_val, self.y_test))
        else:
            raise AttributeError("set must be train, val, test or all")

        class_counts = []
        class_tuples = []

        for i, c in enumerate(classes):
            lsum = np.sum(labels == i)
            class_counts.append(lsum)
            class_tuples.append((classes[i], lsum))

        print(tabulate(class_tuples, headers=('name', 'count')))

        plt.figure()
        plt.title(set + " class distribution")
        plt.bar(np.arange(len(class_counts)), class_counts)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.show()

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(path + "x_test.npy", self.x_test)
        np.save(path + "y_test.npy", self.y_test)
        np.save(path + "x_val.npy", self.x_val)
        np.save(path + "y_val.npy", self.y_val)
        np.save(path + "x_train.npy", self.x_train)
        np.save(path + "y_train.npy", self.y_train)

    @staticmethod
    def load(path:str, class_names:str = None):
        x_test = np.load(path + "x_test.npy")
        y_test = np.load(path + "y_test.npy")
        x_val = np.load(path + "x_val.npy")
        y_val = np.load(path + "y_val.npy")
        x_train = np.load(path + "x_train.npy")
        y_train = np.load(path + "y_train.npy")
        return DataSet(x_train, y_train, x_val, y_val, x_test, y_test, class_names)

    def __str__(self):
        return f"Train set: {self.x_train.shape[0]} samples \nValidation set: {self.x_val.shape[0]} samples \nTest set: {self.x_test.shape[0]} samples"

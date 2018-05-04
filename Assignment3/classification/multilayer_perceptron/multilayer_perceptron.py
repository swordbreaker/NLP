from classification.classifier import Classifier
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.vis_utils import plot_model
import numpy as np
from classification.util.logger import Logger


class MultilayerPerceptron(Classifier):
    """Class for multilayer perceptron"""

    def __init__(self, dataset: "DataSet", num_classes, verbose=0, model=None, epoch=20, logger: "Logger" = None):
        self.epoch = epoch
        self.verbose = verbose
        self.num_classes = num_classes
        if model == None:
            shape = dataset.x_train[0].shape
            model = Sequential()
            model.add(Dense(32, input_shape=shape))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes))
            model.add(Activation('softmax'))
            model.compile(optimizer='sgd', loss='categorical_crossentropy',
                          metrics=[keras.metrics.categorical_accuracy])
            self.classifier = model
        else:
            self.classifier = model
        return super().__init__(dataset, logger=logger)

    def fit(self):
        self.classifier.fit(self.ds.x_train, self.ds.y_train, batch_size=100, epochs=self.epoch, verbose=self.verbose)

    def update(self, x, y):
        self.classifier.fit(x, y, batch_size=100, epochs=self.epoch, verbose=self.verbose)
        self.save_online_model('multilayer_perceptron')

    def validate(self):
        score = self.classifier.evaluate(self.ds.x_test, self.ds.y_test, verbose=self.verbose)
        accuracy = score[1]
        self.logger.log_and_print(f"accuracy: \t {accuracy:04.2f}")
        return accuracy

    def predict(self, x: any) -> [any]:
        predictions = self.classifier.predict_classes(x, verbose=self.verbose)
        n_values = self.num_classes
        return np.eye(n_values)[predictions]

    def predict_proba(self, x):
        return self.classifier.predict_proba(x, verbose=self.verbose)

    def save(self, path: str):
        self.classifier.save_weights(path + 'weights.h5')
        self.classifier.save(path)

    def plot_model(self, path: str = 'data/plots/MultilayerPerceptronModel.png'):
        plot_model(self.classifier, to_file='data/plots/MultilayerPerceptronModel.png')

    @staticmethod
    def load(path: str, dataset: "DataSet") -> "MultilayerPerceptron":
        model = keras.models.load_model(path)
        model.load_weights(path + 'weights.h5')
        sgd = optimizers.SGD(lr=0.01, momentum=0.7, decay=0, nesterov=False)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        return MultilayerPerceptron(dataset, model=model)

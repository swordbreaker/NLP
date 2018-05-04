import numpy as np
from classification.classifier import Classifier
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, GRU, RNN, Masking, Dropout
import keras
import matplotlib.pyplot as plt
import time
from classification.util.logger import Logger


class RnnClassifier(Classifier):
    """Class for gradient boosting"""

    def __init__(self, dataset: "DataSet", n_neurons=100, verbose=2, learning_rate=0.001, logger: "Logger" = None,
                 model: Sequential = None):
        self.history = {}
        self.verbose = verbose

        if model is None:
            n_neurons = 100
            n_features = dataset.y_train.shape[1]
            n_outputs = n_features

            self.model = Sequential([
                Masking(mask_value=0, input_shape=(None, 300)),
                GRU(n_neurons),
                Dropout(0.5),
                Dense(n_outputs),
                Activation('softmax')
            ])

            adam = keras.optimizers.adam(lr=learning_rate)
            self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model = model

        print(self.model.summary())
        return super().__init__(dataset, logger=logger)

    def fit(self, chepoint_path, epochs=100):
        board_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32,
                                                     write_graph=True, write_grads=True,
                                                     write_images=True, embeddings_freq=0, embeddings_layer_names=True,
                                                     embeddings_metadata=True)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(chepoint_path, monitor='val_loss', verbose=0,
                                                                    save_best_only=False, save_weights_only=False,
                                                                    mode='auto', period=5)
        self.history = self.model.fit(self.ds.x_train, self.ds.y_train, epochs=epochs, batch_size=100,
                                      validation_split=0.33, verbose=self.verbose,
                                      callbacks=[board_callback, model_checkpoint_callback])

    def plot_history(self, show_plot=True):
        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if self.logger is not None:
            plt.savefig(self.logger.get_log_path("accuracy", ".png"))
        if show_plot:
            plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if self.logger is not None:
            plt.savefig(self.logger.get_log_path("loss", ".png"))
        if show_plot:
            plt.show()

    def validate(self):
        loss, accuracy = self.model.evaluate(self.ds.x_test, self.ds.y_test, batch_size=128)

        self.logger.log_and_print(f"loss: \t {loss:04.2f}")
        self.logger.log_and_print(f"accuracy: \t {accuracy:04.2f}")

        return accuracy

    def predict(self, x: any) -> [any]:
        p = self.model.predict(x)
        z = np.zeros_like(p)
        z[np.arange(len(p)), p.argmax(1)] = 1
        return z

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def save(self, path: str):
        self.model.save(path)

    @staticmethod
    def load(path: str, dataset: "DataSet", logger: "Logger" = None) -> "RandomForest":
        model = keras.models.load_model(path)
        return RnnClassifier(dataset, model=model, logger=logger)

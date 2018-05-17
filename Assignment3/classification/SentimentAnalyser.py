import random
import numpy
import keras
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional, GRU
from keras.layers import TimeDistributed, Conv2D, Reshape, Masking
from keras.optimizers import Adam
import spacy

from classification import DataSet
import pandas as pd

from classification.classifier import Classifier
import matplotlib.pyplot as plt

import os.path

from IPython.display import clear_output

from sklearn.metrics import precision_recall_fscore_support

class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

    def plot(self, logger):
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.savefig(logger.get_log_path("loss", ".png"))

class SentimentAnalyser(Classifier):
    """description of class"""

    def __init__(self, dataset: "DataSet", n_neurons=16, verbose=1, learning_rate=0.001, max_lenght=300, logger: "Logger" = None,
                model: Sequential = None): 
        self.history = {}
        self.verbose = verbose

        print("Loading spaCy")
        nlp = spacy.load('en_vectors_web_lg')
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        embeddings = nlp.vocab.vectors.data # get embbedings

        if(os.path.isfile("data/data_set32/x_train.npy")):
            dataset = DataSet.load("data/data_set32/", dataset.class_names)
        else:
            self.preprocess(nlp, dataset, max_lenght)
            dataset.save("data_set32/data_set/")

        dataset.y_train = keras.utils.to_categorical(dataset.y_train)
        dataset.y_val = keras.utils.to_categorical(dataset.y_val)
        dataset.y_test = keras.utils.to_categorical(dataset.y_test)

        rnn_shape = {'nr_hidden': n_neurons, 'max_length': max_lenght, 'nr_class': len(dataset.class_names)}
        rnn_settings = {'dropout': 0.5, 'lr': 0.001}

        if model is None:
            self.model = self.compile_rnn(embeddings, rnn_shape, rnn_settings)
        else:
            self.model = model


        print(self.model.summary())
        return super().__init__(dataset, logger=logger)

    def preprocess(self, nlp, ds: "DataSet", max_length):
        print("Parsing texts...")
        train_docs = list(nlp.pipe(ds.x_train))
        val_docs = list(nlp.pipe(ds.x_val))
        test_docs = list(nlp.pipe(ds.x_test))

        train_X = self.get_features(train_docs, max_length)
        val_X = self.get_features(val_docs, max_length)
        test_X = self.get_features(test_docs, max_length)

        ds.x_train = train_X
        ds.x_val = val_X
        ds.x_test = test_X

    def get_features(self, docs, max_length):
        docs = list(docs)
        Xs = numpy.zeros((len(docs), max_length), dtype='int32')
        for i, doc in enumerate(docs):
            j = 0
            for token in doc:
                vector_id = token.vocab.vectors.find(key=token.orth)
                if vector_id >= 0:
                    Xs[i, j] = vector_id
                else:
                    Xs[i, j] = 0
                j += 1
                if j >= max_length:
                    break
        return Xs

    def compile_rnn(self, embeddings, shape, settings):
        model = Sequential()

        model.add(
            Embedding(
                embeddings.shape[0],
                embeddings.shape[1],
                input_length=shape['max_length'],
                trainable=False,
                weights=[embeddings],
                mask_zero=True
            )
        )

        model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
        model.add(Bidirectional(GRU(shape['nr_hidden'],
                                     recurrent_dropout=settings['dropout'],
                                     dropout=settings['dropout'])))
        model.add(Dense(shape['nr_class'], activation='sigmoid'))
        model.compile(optimizer=Adam(lr=settings['lr']), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, chekpoint_path, epochs=100):
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(chekpoint_path + "epoch_{epoch:02d}-val_los_{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=4)

        plot = PlotLearning()

        self.model.fit(self.ds.x_train, self.ds.y_train, validation_data=(self.ds.x_val, self.ds.y_val),
                          epochs=epochs, batch_size=500, callbacks=[model_checkpoint_callback, plot], verbose=self.verbose)

        plot.plot(self.logger)

    def validate(self):
        loss, accuracy = self.model.evaluate(self.ds.x_test, self.ds.y_test, batch_size=128)

        self.logger.log_and_print(f"loss: \t {loss:04.2f}")
        self.logger.log_and_print(f"accuracy: \t {accuracy:04.2f}")

        return accuracy

    def predict(self, x: any) -> [any]:
        p = self.model.predict(x)
        z = numpy.zeros_like(p)
        z[numpy.arange(len(p)), p.argmax(1)] = 1
        return z

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def save(self, path: str):
        self.model.save(path)


    def metrics_task2(self):
        y_true = self.ds.y_test
        y_pred = self.predict(self.ds.x_test)

        y_true = [one_hot[3] + one_hot[4] for one_hot in y_true]
        y_pred = [one_hot[3] + one_hot[4] for one_hot in y_pred]
        
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

    def metrics_task3(self):
        y_true = self.ds.y_test
        y_pred = self.predict(self.ds.x_test)

        y_true = [one_hot[1] + one_hot[2] + one_hot[3] + one_hot[4] for one_hot in y_true]
        y_pred = [one_hot[1] + one_hot[2] + one_hot[3] + one_hot[4] for one_hot in y_pred]
        
        #(precision, recall, fscore, _) = precision_recall_fscore_support(y_true, y_pred, average='weighted')
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

    @staticmethod
    def load(path: str, dataset: "DataSet", logger: "Logger" = None) -> "RandomForest":
        model = keras.models.load_model(path)
        return SentimentAnalyser(dataset, model=model, logger=logger)


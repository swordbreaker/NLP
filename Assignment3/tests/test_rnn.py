import numpy as np
import itertools
from classification.RNN.rnn_classifier import RnnClassifier
from classification.data_set import DataSet
from sklearn import preprocessing
from classification.ticketing_data import *
import keras
import os.path
import time
from classification.util.logger import Logger

labels, class_names = get_merged_labels_three(root='../')

x = get_fast_text_tickets_message(root='../')
y = labels

max = 0
l = []

for words in x:
    n = words.shape[0]
    if n > max:
        max = n

new_x = np.zeros((x.shape[0], max, x[0].shape[1]))

i = 0
for words in x:
    new_x[i, :words.shape[0], :] = words
    i += 1

x = new_x

y = keras.utils.to_categorical(y)

data_set = DataSet.from_np_array(x, y, class_names=class_names)

path = "../classification/RNN/saved_model/rnn.model"

with Logger("rnn", root='../') as l:
    l.log_and_print(data_set)
    l.log("")

    if os.path.isfile(path):
        classifier = RnnClassifier.load(path, data_set, logger=l)
    else:
        classifier = RnnClassifier(data_set, logger=l)

    classifier.fit(path, epochs=20)

    classifier.validate()
    classifier.metrics()
    classifier.plot_confusion_matrix()
    classifier.plot_history()
    classifier.save(path)

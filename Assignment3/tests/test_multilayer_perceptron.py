import numpy as np
import itertools
from classification.multilayer_perceptron import *
from classification.data_set import DataSet
from sklearn import preprocessing
from classification.ticketing_data import *
from classification.util.logger import Logger

labels, class_names = get_merged_labels_three(root='../')

x = get_doc_vec_ticketing_message(root='../')
y = labels

n_values = len(class_names)
y = np.eye(n_values)[y]

data_set = DataSet.from_np_array(x, y, class_names=class_names, p_train=0.8, p_val=0.1)

with Logger("multilayer_perceptron", root='../') as l:
    l.log_and_print(data_set)
    l.log("")

    # classifier = multilayer_perceptron.MultilayerPerceptron(data_set, num_classes=len(class_names), epoch=50, verbose=1,
    #                                                        logger=l)
    # classifier.fit()
    # classifier.validate()
    # classifier.metrics()
    # classifier.plot_confusion_matrix()

    model = hyperparameter_tuning.fit_hyper(root='../')
    classifier = multilayer_perceptron.MultilayerPerceptron(data_set, num_classes=len(class_names), epoch=20, verbose=0,
                                                            model=model, logger=l)
    classifier.validate()
    classifier.metrics()
    classifier.plot_confusion_matrix()

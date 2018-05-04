import numpy as np
import itertools
from classification.svm.svm import SupportingVectorMachine
from classification.data_set import DataSet
from sklearn import preprocessing
from classification.ticketing_data import *
from classification.util.logger import Logger

labels, class_names = get_merged_labels_three(root='../')
raw_data = load_messages(root='../')

x = get_doc_vec_ticketing_message(root='../')
y = labels

# twitter data
# class_names = ['positive', 'negative', 'neutral']
# x = np.load("data/test_data/fastTextDocumentVector.npy")
# positv 0, negative 1, neutral 3
# y = np.load("data/test_data/labels.npy")

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

data_set = DataSet.from_np_array(x, y, class_names=class_names, raw_data=raw_data, p_train=0.8, p_val=0.1)
'''
with Logger("svm", root='../') as l:
    l.log_and_print(data_set)
    l.log_and_print()

    classifier = SupportingVectorMachine(data_set, verbose=1, logger=l)
    classifier.hyper_parameter_tuning()
    classifier.validate()
    classifier.metrics()
    #  classifier.print_wrong_test()
    classifier.plot_confusion_matrix()
    classifier.save('../data/saved_models/svm_optimised.pkl')
'''

#  Train password classifier
labels, class_names = get_password_data(root='../')
x = get_doc_vec_ticketing_message(root='../')
y = labels
le.fit(y)
y = le.transform(y)

data_set = DataSet.from_np_array(x, y, class_names=class_names, p_train=0.8, p_val=0.1)
with Logger("svm", root='../') as l:
    l.log_and_print("Password classifier")
    l.log_and_print(data_set)
    l.log_and_print()

    classifier = SupportingVectorMachine(data_set, verbose=1, logger=l)
    classifier.hyper_parameter_tuning()
    classifier.validate()
    classifier.metrics()
    classifier.plot_confusion_matrix()
    classifier.save('../data/saved_models/svm_password.pkl')

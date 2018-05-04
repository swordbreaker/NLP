import numpy as np
import itertools
from classification.gradien_boost.gradien_boost_classifier import GradienBoost
from classification.data_set import DataSet
from sklearn import preprocessing
from classification.ticketing_data import *
from classification.util.logger import Logger

labels, class_names = get_merged_labels_three(root='../')

x = get_doc_vec_ticketing_message(root='../')
y = labels

# twitter data
# class_names = ['positive', 'negative', 'neutral']
# x = np.load("data/test_data/fastTextDocumentVector.npy")
## positv 0, negative 1, neutral 3
# y = np.load("data/test_data/labels.npy")


le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

data_set = DataSet.from_np_array(x, y, class_names=class_names, p_train=0.8, p_val=0.1)
data_set.plot_distribution('train')
data_set.plot_distribution('val')
data_set.plot_distribution('test')
data_set.plot_distribution('all')


with Logger("gradient_boost", root='../') as l:
    l.log_and_print(data_set)
    l.log("")

    classifier = GradienBoost(data_set, n_estimators=120, verbose=1, logger=l)
    # classifier.hyper_parameter_tuning()
    classifier.fit()
    classifier.validate()
    classifier.metrics()
    classifier.plot_confusion_matrix()

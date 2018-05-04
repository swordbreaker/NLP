import numpy as np
import itertools
from classification.svm.svm import SupportingVectorMachine
from classification.voting.voting_classifier import Voting
from classification.data_set import DataSet
from sklearn import preprocessing
from classification.ticketing_data import *
from classification.util.logger import Logger
from classification.gradien_boost.gradien_boost_classifier import GradienBoost
from classification.random_forest.random_forest_classifier import RandomForest

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

with Logger("voting", root='../') as l:
    l.log_and_print(data_set)
    l.log_and_print()

    svm = SupportingVectorMachine(data_set, verbose=0, logger=l)
    gradient_boost = GradienBoost(data_set, verbose=0, n_estimators=120, logger=l)
    random_forest = RandomForest(data_set, verbose=0, logger=l)

    svm.fit()
    gradient_boost.fit()
    random_forest.fit()

    classifier = Voting(data_set, estimators=[('svm', svm.classifier), ('gradient_boost', gradient_boost.estimator),
                                              ('random_forest', random_forest.estimator)], logger=l, voting="soft")
    classifier.fit()

    l.log_and_print("SVM")
    svm.validate()
    svm.metrics()
    l.log_and_print()

    l.log_and_print("gradient_boost")
    gradient_boost.validate()
    gradient_boost.metrics()
    l.log_and_print()

    l.log_and_print("random_forest")
    random_forest.validate()
    random_forest.metrics()
    l.log_and_print()

    l.log_and_print("Voting")
    classifier.validate()
    classifier.metrics()
    classifier.plot_confusion_matrix()

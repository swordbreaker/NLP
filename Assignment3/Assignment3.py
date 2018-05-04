import spacy as sp
import numpy as np 
import pandas as pd
import os.path
import sys
import math

from classification import DataSet
from classification import *

# python -m spacy download de_core_news_sm


def load_document():
    """ returns a dataframe with the collums 'sentimens', 'product', 'review'"""
    df = pd.read_csv("Amazon_Baby_train.txt", sep=';', header=None)
    df.columns = ['sentimens', 'product', 'review']
    return df


def load_doc_vec(df: pd.DataFrame):
    """ returns tuple with
            document vectors with the shape (n_samples, doc_vec_dimention)
            labels with the shape (n_samples)
    """
    nlp = sp.load("de_core_news_sm")
    tokens = nlp(df['review'][0])
    doc_vec = tokens.vector

    x = np.zeros((df.shape[0], doc_vec.shape[0]))
    y = np.array(df['sentimens'], dtype=np.int32)

    if(os.path.isfile("data/x_document.npy") and os.path.isfile("data/y_document.npy")):
        x = np.load("data/x_document.npy")
        y = np.load("data/y_document.npy")
    else:
        for i in range(df.shape[0]):
            tokens = nlp(df['review'][i])
            x[i,:] = tokens.vector
        np.save("data/x_document.npy", x)
        np.save("data/y_document.npy", y)
    return x, y

def load_word_vecs(df: pd.DataFrame, max_lenght = 5000):
    """ returns tuple with
            word vectors with the shape (n_samples, max_lenght, word_vec_dimention)
            labels with the shape (n_samples)
    """
    nlp = sp.load("de_core_news_sm")
    tokens = nlp(df['review'][0])
    word_vec = tokens[0].vector

    x = np.zeros((df.shape[0], max_lenght, word_vec.shape[0]))
    y = np.array(df['sentimens'], dtype=np.int32)

    if(os.path.isfile("data/x_words.npy") and os.path.isfile("data/y_words.npy")):
        x = np.load("data/x_words.npy")
        y = np.load("data/y_words.npy")
    else:
        for i in range(df.shape[0]):
            tokens = nlp(df['review'][i])

            l = min((len(tokens), max_lenght))
            for k in range(l):
                token = tokens[k]
                x[i,k,:] = token.vector
        np.save("data/x_document.npy", x)
        np.save("data/y_document.npy", y)
    return x, y

df = load_document()
x, y = load_doc_vec(df)

ds = DataSet.from_np_array(x, y, class_names=[1,2,3,4,5], p_train=0.8, p_val=0.1, shuffle=True)
#ds.plot_distribution('train')
#ds.plot_distribution('val')
#ds.plot_distribution('test')
#ds.plot_distribution('all')

with Logger("svm", root='') as l:
    l.log_and_print(ds)
    l.log("")

    #classifier = GradienBoost(ds, n_estimators=120, verbose=1, logger=l)
    classifier = SupportingVectorMachine(ds, logger=l, verbose=1)
    # classifier.hyper_parameter_tuning()
    classifier.fit()
    classifier.validate()
    classifier.metrics()
    classifier.plot_confusion_matrix()
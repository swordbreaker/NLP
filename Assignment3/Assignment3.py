import spacy as sp
import numpy as np 
import pandas as pd
import os.path
import sys
import math
import os

from classification import DataSet
from classification import *

from classification.SentimentAnalyser import SentimentAnalyser
# !python -m spacy download en_core_web_sm

def load_document():
    """ returns a dataframe with the collums 'sentimens', 'product', 'review'"""
    df = pd.read_csv("Amazon_Baby_train.txt", sep=';', header=None)
    df.columns = ['sentimens', 'product', 'review']
    #df = df[(df.sentimens == 1.0) | (df.sentimens == 2.0) | (df.sentimens == 3.0) | (df.sentimens == 4.0) | (df.sentimens == 5.0)]
    df.sentimens = df.sentimens - 1
    return df


def load_doc_vec(df: pd.DataFrame):
    """ returns tuple with
            document vectors with the shape (n_samples, doc_vec_dimention)
            labels with the shape (n_samples)
    """

    if(os.path.isfile("data/x_document.npy") and os.path.isfile("data/y_document.npy")):
        x = np.load("data/x_document.npy")
        y = np.load("data/y_document.npy")
    else:
        nlp = sp.load("en_core_web_sm")

        docs = list(nlp.pipe(df['review']))
        x = np.zeros((df.shape[0], docs[0].vector.shape[0]))
        y = np.array(df['sentimens'], dtype=np.int32)

        for i, doc in enumerate(docs):
            x[i,:] = doc.vector

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

def rnn():

    df = load_document()
    ds = DataSet.from_np_array(df['review'], np.asarray(df['sentimens'], dtype='int32'), class_names=[1,2,3,4,5], shuffle=True, p_train=0.9, p_val=0.05)
    ds.plot_distribution('train')
    ds.plot_distribution('val')
    ds.plot_distribution('test')
    ds.plot_distribution('all')

    #disable cuda
    #import os
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    path = "checkpoints/best.hdf5"

    with Logger("rnn", root='') as l:
        l.log_and_print(ds)
        l.log("")

        if os.path.isfile(path):
            classifier = SentimentAnalyser.load(path, ds, logger=l)
        else:
            classifier = SentimentAnalyser(ds, logger=l)

        classifier.fit("checkpoints/", 2)
        classifier.validate()
        classifier.metrics()
        classifier.plot_confusion_matrix()

def simple():

    df = load_document()
    x, y = load_doc_vec(df)

    ds = DataSet.from_np_array(x, y, class_names=[1,2,3,4,5], p_train=0.8, p_val=0.1, shuffle=True)

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

import classification.emotion_analizer as emotion_analizer

def emotionality():
    pd = load_document()
    texts = list(pd['review'])
    texts = [text for text in texts if text.strip() != '']
    best = emotion_analizer.findBestEmotional(texts)
    print([(doc.user_data['emotionality'], doc.text) for doc in best[:5]])

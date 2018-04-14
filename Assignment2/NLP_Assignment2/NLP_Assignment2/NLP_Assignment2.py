import sys
import nltk
import math
import itertools
import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
from typing import Callable
from nltk.corpus import brown
%matplotlib inline

mycorpus = nltk.corpus.reader.TaggedCorpusReader("", "POS_German_train.txt")

print(mycorpus.tagged_words())


raw = 'I do not like green eggs and ham.'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NOUN')


brown_tagged_sents = brown.tagged_sents(categories='news', tagset='universal')
brown_sents = brown.sents(categories='news')

patterns = [
    (r'.*ing$', 'VERB'),  # gerunds
    (r'.*ed$', 'VERB'),  # simple past
    (r'.*es$', 'VERB'),  # 3rd singular present
    (r'.*ould$', 'VERB'),  # modals
    (r'and', 'CONJ'),  # and
    (r'to', 'PRT'),  # to
    (r'^th.*', 'DET'),  # the
    (r'o[nf]', 'ADP'),  # of,on
    (r'[.,;!?\'`]', '.'),  # . and , etc.
    (r'.*\'s$', 'NOUN'),  # possessive nouns
    (r'.*s$', 'NOUN'),  # plural nouns
    (r'^‐?[0‐9]+(.[0‐9]+)?$', 'NUM'),  # cardinal numbers
    (r'.*ly$', 'ADV'),  # adverbs
    (r'.*', 'NOUN')  # nouns (default)
]



regexp_tagger = nltk.RegexpTagger(patterns)

fd = nltk.FreqDist(brown.words(categories='news'))  # frequency distribution of words

cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news', tagset='universal'))
likely_100_tags = dict((word, cfd[word].max()) for (word, _) in fd.most_common(10000))

baseline_tagger = nltk.UnigramTagger(model=likely_100_tags,  # assign 'NOUN' for OOV words
                                     backoff=nltk.DefaultTagger('NOUN'))


# Splitt data set
size = int(len(brown.words(categories='news')) * 0.8)

train_set = brown.tagged_words(categories='news')[:size]
eval_set = brown.tagged_words(categories='news')[size:]

# Splitt data set
size = int(len(brown_tagged_sents) * 0.8)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

t0 = nltk.DefaultTagger('NOUN')
t1 = nltk.RegexpTagger(patterns, backoff=t0)
t2 = nltk.UnigramTagger(model=likely_100_tags, backoff=t1)
t3 = nltk.UnigramTagger(train_sents, backoff=t2)
t4 = nltk.BigramTagger(train_sents, backoff=t3)

# t0 = nltk.DefaultTagger('NN')
# t1 = nltk.UnigramTagger(train_sents, backoff=t0)
# t3 = nltk.UnigramTagger(model=likely_100_tags, backoff=t1)
# t4 = nltk.BigramTagger( train_sents, backoff=t3)
t_final = t4

print("accuracy of Bigram  Tagger: ", t_final.evaluate(test_sents))
# print("pos tag: ", nltk.pos_tag(test_sents) )



test_tags = [tag for sent in brown.sents(categories='editorial') for (word, tag) in t_final.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial', tagset='universal')]

print(nltk.ConfusionMatrix(gold_tags, test_tags))
print(t_final.evaluate(brown.tagged_sents(categories='editorial', tagset='universal')))

pos_tag = nltk.pos_tag(brown.words(categories='editorial'), tagset='universal')
test_tags = [tag for (word, tag) in pos_tag]
conf_mat = nltk.ConfusionMatrix(gold_tags, test_tags)
print(conf_mat)
print(nltk.accuracy(gold_tags, test_tags))
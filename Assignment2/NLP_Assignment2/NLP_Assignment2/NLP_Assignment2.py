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
import re
# %matplotlib inline


class POSTagger():
    # mapping Stuttgart-Tübingen-Tagset STTS to Universal Part-of-Speech Tagset
    mapping = {
        'ADJA': 'ADJ',
        'ADJD': 'ADJ',
        'ADV': 'ADV',
        'APPR': 'ADP',
        'APPRART': 'ADP',
        'APPO': 'ADP',
        'APZR': 'ADP',
        'ART': 'DET',
        'CARD': 'NUM',
        'FM': 'X',
        'ITJ': 'X',
        'KOUI': 'CONJ',
        'KOUS': 'CONJ',
        'KON': 'CONJ',
        'KOKOM': 'CONJ',
        'NN': 'NOUN',
        'NE': 'NOUN',
        'PDS': 'PRON',
        'PDAT': 'PRON',
        'PIS': 'PRON',
        'PIAT': 'PRON',
        'PIDAT': 'PRON',
        'PPER': 'PRON',
        'PPOSS': 'PRON',
        'PPOSAT': 'PRON',
        'PRELS': 'PRON',
        'PRELAT': 'PRON',
        'PRF': 'PRON',
        'PWS': 'PRON',
        'PWAT': 'PRON',
        'PWAV': 'PRON',
        'PAV': 'ADV',
        'PTKZU': 'PRT',
        'PTKNEG': 'PRT',
        'PTKVZ': 'PRT',
        'PTKANT': 'PRT',
        'PTKA': 'PRT',
        'SGML': 'X',
        'SPELL': 'X',
        'TRUNC': 'PRT',
        'VVFIN': 'VERB',
        'VVIMP': 'VERB',
        'VVINF': 'VERB',
        'VVIZU': 'VERB',
        'VVPP': 'VERB',
        'VAFIN': 'VERB',
        'VAIMP': 'VERB',
        'VAINF': 'VERB',
        'VAPP': 'VERB',
        'VMFIN': 'VERB',
        'VMINF': 'VERB',
        'VMPP': 'VERB',
        'XY': 'X',
        '$,': ',',
        '$.': '.',
        '$(': 'X',
        }

    regex_patterns = [
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


    def __init__(self, path):
        self.tagged_sents, self.sents, self.tagged_words, self.words = self.__preprocessing(path)

        self.tagged_sents_train = []
        self.tagged_sents_test = []
        self.sents_train = []
        self.sents_test = []
        self.tagged_words_train = []
        self.tagged_words_test = []

        self.path = path
        self.currentTagger = nltk.DefaultTagger('NOUN')
        
        self.__splitt_set()

    def __preprocessing(self, path):
        """ returns tagged_sents, sents, tagged_word, words """
        with open(path) as f:
            tagged_sents = []
            sents = []
            tagged_words = []
            words = []

            content = f.readlines()
    
            for line in content:
                tagged_sents.append([])
                sents.append([])
                line = line.strip()
                line = re.sub(";$", '', line)
       
                for word, tag in re.findall('([^\s;]+)/([\w$(,.]+)', line):

                    if(tag not in self.mapping):
                        print(f"cannot find tag {t} in mapping")
                        continue

                    tagged_sents[-1].append((word, self.mapping[tag]))
                    sents[-1].append(word)
                    tagged_words.append((word, self.mapping[tag]))
                    words.append(word)
            
            return (tagged_sents, sents, tagged_words, words)

    def __splitt_set(self):
        p_train = 0.8

        n = len(self.tagged_sents)
        n_train = int(n * p_train)

        self.tagged_sents_train = self.tagged_sents[:n_train]
        self.tagged_sents_test   = self.tagged_sents[n_train:]

        self.sents_train = self.sents[:n_train]
        self.sents_test = self.sents[n_train:]

        self.tagged_words_train = self.tagged_words[:n_train]
        self.tagged_words_test = self.tagged_words[n_train:]

    def get_regex_tagger(self):
        return nltk.RegexpTagger(self.regex_patterns)

    def get_likely_m_tags(self, n : int = 10000):
        fd = nltk.FreqDist(self.words)  # frequency distribution of words
        cfd = nltk.ConditionalFreqDist(self.tagged_words)
        return dict((word, cfd[word].max()) for (word, _) in fd.most_common(n))

    def add_regex_tagget(self):
        tagger = nltk.RegexpTagger(self.regex_patterns, backoff=self.currentTagger)
        self.currentTagger = tagger
        return self

    def add_likely_n_tags_tagger(self, n):
        tagger = nltk.UnigramTagger(model=self.get_likely_m_tags(n), backoff=self.currentTagger)
        self.currentTagger = tagger
        return self

    def add_unigram_tagger(self):
        tagger = nltk.UnigramTagger(self.tagged_sents_train, backoff=self.currentTagger)
        self.currentTagger = tagger
        return self

    def add_bigram_tagger(self):
        tagger = nltk.BigramTagger(self.tagged_sents_train, backoff=self.currentTagger)
        self.currentTagger = tagger
        return self

    def add_trigram_tagger(self):
        tagger = nltk.TrigramTagger(self.tagged_sents_train, backoff=self.currentTagger)
        self.currentTagger = tagger
        return self

    def use_hmm_tagger(self):
        symbols = set(self.words)
        tag_set = set([tag for word, tag in self.tagged_words])
        tagger = nltk.HiddenMarkovModelTrainer(tag_set, symbols)
        tagger = tagger.train_supervised(self.tagged_sents_train)
        self.currentTagger = tagger
        return self


    def eval(self):
        print("accuracy train: ", self.currentTagger.evaluate(self.tagged_sents_train))
        print("accuracy test: ", self.currentTagger.evaluate(self.tagged_sents_test))

    def test(self, path):
        tagged_sents, sents, tagged_words, words = self.__preprocessing(path)
        test_tags = [tag for sent in sents for (word, tag) in self.currentTagger.tag(sent)]
        gold_tags = [tag for (word, tag) in tagged_words]
        print(nltk.ConfusionMatrix(gold_tags, test_tags))
        print("accuracy eval: ", self.currentTagger.evaluate(tagged_sents))

tagger = POSTagger('POS_German_train.txt')

tagger.add_regex_tagget()
tagger.add_likely_n_tags_tagger(100)
tagger.add_unigram_tagger()
tagger.add_bigram_tagger()
tagger.add_trigram_tagger()
#tagger.use_hmm_tagger()

tagger.eval()
tagger.test("POS_German_minitest.txt")
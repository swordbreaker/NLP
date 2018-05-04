from classification.data_set import DataSet
from sklearn import preprocessing
from classification.ticketing_data import *
from classification.util.logger import Logger
from classification.svm.sgd import StochastikGradientDescentClassifier
import re
import pandas as pd


def filter_ticket_messages(messages):
    header_re = re.compile('^.*-{5,} Betreff:.*?:')
    header2_re = re.compile('^.*Ihr Anliegen')
    header3_re = re.compile('^.*Betreff: .* Problembeschrieb:')
    footer_re = re.compile('Database Log info: .*$')
    signatur_re = re.compile(
        '(-{5,}|_{5,}) (Fachhochschule Nordwestschweiz)?.*? (-{5,}|_{5,})( .*?(-{5,}|_{5,})((Anw.*?($))?(-{5,})?)?)?')
    mail_answer_re = re.compile('Von: .*? Betreff:')
    filtered = []
    for message in messages:
        message = header_re.sub(' ', message)
        message = header2_re.sub(' ', message)
        message = header3_re.sub(' ', message)
        message = footer_re.sub(' ', message)
        message = signatur_re.sub(' ', message)
        message = mail_answer_re.sub(' ', message)
        filtered.append(message)
    return filtered


labels, class_names = get_merged_labels_three(root='../')

x = get_doc_vec_ticketing_message(root='../')
y = labels
print(x.shape)

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

df = pd.read_csv('../data/trainingData/ticketDump.csv', sep=';', encoding='UTF-8', na_filter=False)
categories = ['"FHNW Benutzerpasswort von Studierenden zurücksetzen"',
                  '"FHNW Passwortänderung (Active Directory)"',
                  '"VPN Zugriff"', '"Drucker technische Probleme"',
                  '"Drucker verbinden"', '"Webmail technische Probleme"', '"Papierstau"',
                  '"VPN technische Probleme"', '"Webmail Zugriff"',
                  '"SWITCHengines - Cloud Infrastructure"', '"Datenablage"', '"Web Single Sign On AAI"',
                  '"Benutzendenkonto gesperrt"', '"Speicherplatz"', '"Benutzername vergessen"', '"Passwort ändern"']
df = df.loc[df['category'].isin(categories)]
sentences = filter_ticket_messages(df.message)

data_set = DataSet.from_np_array(x, y, class_names=class_names, p_train=0.8, p_val=0.1)
data_set.add_text_data(sentences)

with Logger("stochastik_gradient_descend", root='../') as l:
    l.log_and_print(data_set)
    l.log_and_print()

    classifier = StochastikGradientDescentClassifier(data_set, logger=l)
    l.log_and_print("SGD with document vector")
    #classifier.hyperparameter()
    #classifier.validate()
    #classifier.metrics()
    #classifier.plot_confusion_matrix()
    l.log_and_print()
    l.log_and_print("SGD with tf idf")
    classifier.hyperparameter_tf()
    classifier.validate()
    classifier.metrics()
    classifier.plot_confusion_matrix()

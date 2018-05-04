import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
from scipy import sparse


def get_doc_vec_ticketing_message(root: str = ""):
    """
        shape: 517 x 300
        content: document vector which is the mean of all word vectors
    """
    return np.load(root + "data/test_data/fastTextDocTicketsMessage.npy")


def get_doc_vec_ticketing_subject(root: str = ""):
    """
        shape: 517 x 300
        content: document vector which is the mean of all word vectors
    """
    return np.load(root + "data/test_data/fastTextDocTicketsSubject.npy")


def get_fast_text_tickets_message(root: str = ""):
    """
        shape:  517
        content: [[w1], [w2], .., wn] where wn is an vector with 300 dimensions
    """
    return np.load(root + "data/test_data/fastTextTicketsMessage.npy")


def get_fast_text_tickets_subject(root: str = ""):
    """
        shape: 517
        conent: [[w1], [w2], .., wn] where wn is an vector with 300 dimensions
    """
    return np.load(root + "data/test_data/fastTextTicketingSubject.npy")


def get_tfidf_tickets_message(root: str = ""):
    """
        shape: 517
        content: array with shape (word_count, 2) the 2nd dimesion contains (#occurrence, TF-IDF value))
    """
    return sparse.load_npz(root + "data/test_data/tfidfTicketsMessage.npz")


def get_tfidf_subject(root: str = ""):
    """
        shape: 517 
        content: array with shape (word_count, 2) the 2nd dimesion contains (#occurrence, TF-IDF value))
    """
    return sparse.load_npz(root + "data/test_data/tfidfTicketsSubject.npz")


def get_w2v_tickets_message(root: str = ""):
    """
        shape: 517
        content: [[w1], [w2], .., wn] where wn is an vector with 300 dimensions
    """
    return np.load(root + "data/test_data/w2vTicketsMessage.npy")


def get_w2v_tickets_subject(root: str = ""):
    """
        shape: 517 
        content: [[w1], [w2], .., wn] where wn is an vector with 300 dimensions
    """
    return np.load(root + "data/test_data/w2vTicketsSubject.npy")


def get_ticketing_labels(root: str = ""):
    """
        shape: (7135,)
        0  FHNW Benutzerpasswort von Studierenden zurücksetzen
        1  FHNW Passwort änderung (Active Directory)
        2  VPN Zugriff
        3  Drucker technische Probleme
        4  Drucker verbinden
        5  Webmail technische Probleme
        6  Papierstau
        7  VPN technische Probleme
        8  Webmail Zugriff
        9  SWITCHengines - Cloud Infrastructure
        9  Datenablage
        9  Web Single Sign On AAI
        9  Speicherplatz
        10 Benutzerkonto gesperrt
        11 Benutzername vergessen
        12 Passwort ändern
    """
    return np.load(root + "data/test_data/ticketing_labels.npy")


def get_ticketing_class_names():
    return ['FHNW Benutzerpasswort von Studierenden zurücksetzen', 'FHNW Passwort änderung (Active Directory)',
            'VPN Zugriff', 'Drucker technische Probleme', 'Drucker verbinden', 'Webmail technische Probleme',
            'Papierstau', 'VPN technische Probleme', 'Webmail Zugriff', 'Andere',
            'Benutzerkonto gesperrt', 'Benutzername vergessen', '"Passwort ändern"']


def merge_labels(merge_dict: dict, root: str = ""):
    labels = get_ticketing_labels(root)
    n = labels.shape[0]
    for i in range(n):
        labels[i] = merge_dict[labels[i]]

    return labels


def get_merged_labels_one(root: str = ""):
    label_dict = {
        0: 0,  # login
        1: 0,
        2: 1,  # vpn
        3: 2,  # drucker
        4: 2,
        5: 3,  # webmail
        6: 2,  # drucker
        7: 1,  # vpn
        8: 3,  # webmail
        9: 4,  # andere
        10: 0,  # login
        11: 0,  # login
        12: 0  # login
    }

    class_names = ['login', 'vpn', 'drucker', 'webmail', 'andere']
    labels = merge_labels(label_dict, root)
    np.random.seed(42)
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    labels = labels[idx]
    return labels, class_names


def get_merged_labels_two(root: str = ""):
    label_dict = {
        0: 0,  # login
        1: 0,
        2: 1,  # vpn
        3: 2,  # drucker
        4: 2,
        5: 3,  # andere
        6: 2,  # drucker
        7: 1,  # vpn
        8: 3,  # andere
        9: 3,
        10: 0,  # login
        11: 0,  # login
        12: 0  # login
    }

    class_names = ['login', 'vpn', 'drucker', 'andere']
    labels = merge_labels(label_dict, root)
    np.random.seed(42)
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    labels = labels[idx]
    return labels, class_names


def get_merged_labels_three(root: str = ""):
    label_dict = {
        0: 0,  # login
        1: 0,  # login
        2: 1,  # vpn
        3: 2,  # andere
        4: 2,  # andere
        5: 2,  # andere
        6: 2,  # andere
        7: 1,  # vpn
        8: 2,  # andere
        9: 2,  # andere
        10: 0,  # login
        11: 0,  # login
        12: 0  # login
    }

    class_names = ['login', 'vpn', 'andere']
    labels = merge_labels(label_dict, root)
    np.random.seed(42)
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    labels = labels[idx]
    return labels, class_names


def get_password_data(root: str = ''):
    label_dict = {
        0: 0,  # reset password
        1: 0,  # reset password
        2: 4,  # other
        3: 4,  # other
        4: 4,  # other
        5: 4,  # other
        6: 4,  # other
        7: 4,  # other
        8: 4,  # other
        9: 4,  # other
        10: 2,  # locked
        11: 3,  # username forgotten
        12: 1  # change password
    }
    class_names = ['passwort-zuruecksetzen', 'passwort-aendern', 'gesperrt', 'username-vergessen', 'andere']
    labels = merge_labels(label_dict, root)
    np.random.seed(42)
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    labels = labels[idx]
    return labels, class_names


def plot_distribution():
    classes = get_ticketing_class_names()
    labels = get_ticketing_labels()

    class_counts = []
    class_tuples = []

    for i, c in enumerate(classes):
        lsum = np.sum(labels == i)
        class_counts.append(lsum)
        class_tuples.append((classes[i], lsum))

    print(tabulate(class_tuples, headers=('name', 'count')))

    plt.figure()
    plt.bar(np.arange(len(class_counts)), class_counts)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.show()


def load_messages(root='') -> [str]:
    ''' Loads the raw message strings from a .csv file '''
    df = pd.read_csv(root + 'data/test_data/messagesTicketDump.csv', encoding='UTF-8')
    return df.message


def load_subjects(root='') -> [str]:
    ''' Loads the raw subject strings from a .csv file '''
    df = pd.read_csv(root + 'data/test_data/subjectsTicketDump.csv', encoding='UTF-8')
    return df.subject

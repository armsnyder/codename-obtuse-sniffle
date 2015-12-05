__author__ = 'flame and also fawkes'

import os
import pickle


def load_pickle(filename):
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))
    return []


def save_pickle(object, filename):
    model_file = open(filename, 'wb')
    pickle.dump(object, model_file)
    model_file.close()

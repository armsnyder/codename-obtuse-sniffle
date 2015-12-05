__author__ = 'flame'

import matplotlib.pyplot as plt
import numpy as np
import library
from mnist import load_mnist
import random


def look_through_sevens(k):
    images, labels = load_mnist(digits=[7], path='.')
    # Displaying the mean image for digit 9.
    indices = random.sample(range(len(images)), k)
    for i in indices:
        plt.imshow(images[i], cmap='gray')
        plt.show()


def generate_training_and_testing_sets():
    training_set_1 = []
    training_set_2 = []
    training_set_3 = []
    testing_set = []
    for digit in xrange(10):
        images, labels = load_mnist(digits=[digit], path='.')
        training_indices = random.sample(range(len(images)), 1100)
        testing_indices = random.sample(training_indices, 100)
        training_indices = [x for x in training_indices if x not in testing_indices]
        if digit in [0, 1, 2]:
            training_set_1.extend([(images[i], labels[i]) for i in training_indices])
        if digit in [3, 4, 5]:
            training_set_2.extend([(images[i], labels[i]) for i in training_indices])
        if digit in [6, 7, 8, 9]:
            training_set_3.extend([(images[i], labels[i]) for i in training_indices])
        testing_set.extend([(images[i], labels[i]) for i in testing_indices])
    library.save_pickle(training_set_1, 'training_set_1.p')
    library.save_pickle(training_set_2, 'training_set_2.p')
    library.save_pickle(training_set_3, 'training_set_3.p')
    library.save_pickle(testing_set, 'testing_set.p')


if __name__ == '__main__':
    generate_training_and_testing_sets()
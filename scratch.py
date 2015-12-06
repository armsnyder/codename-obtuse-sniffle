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


def plot_a_thing(thing):
    # plt.bar(range(len(thing)), [x[1] for x in thing])
    plt.plot([x[0] for x in thing], [x[1] for x in thing], 'o')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlim((0.00007, 13))
    plt.title('SVM Error vs Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Error')
    # plt.xticks([x+0.4 for x in range(len(thing))], [x[0][1] for x in thing])
    # plt.ylim((0, 1))
    plt.show()


def problem_5b2(input):
    results = []
    # Code for loading data
    print 'loading data'
    training_set, training_labels, testing_set, testing_labels = (
        pickle.load(open('training_set_2.p')),
        pickle.load(open('training_labels_2.p')),
        pickle.load(open('testing_set_2.p')),
        pickle.load(open('testing_labels_2.p')))
    # preprocessing
    print 'preprocessing data'
    training_set = preprocess(training_set, False)
    testing_set = preprocess(testing_set, False)
    for trial in input:
        print "TRIAL", trial
        # build_classifier is a function that takes in training data and outputs an sklearn classifier.
        print 'building classifier'
        classifier = build_classifier(training_set, training_labels, 8, 'rbf', 3, trial)
        print 'predicting'
        predicted = classify(testing_set, classifier)
        error = error_measure(predicted, testing_labels)
        print '  error:', error
        results.append(error)
    return zip(input, results)


def problem_5b(training_set_sizes, testing_set_size=None):
    results = []
    if testing_set_size:
        for training_set_size in training_set_sizes:
            print "Starting", training_set_size

            print 'loading data'
            training_set, training_labels, testing_set, testing_labels = select_data(training_set_size,
                                                                                     testing_set_size)
            testing_set = pickle.load(open('testing_set_2.p'))
            testing_labels = pickle.load(open('testing_labels_2.p'))

            print 'preprocessing data'
            training_set = preprocess(training_set, False)
            testing_set = preprocess(testing_set, False)

            print 'building classifier'
            classifier = build_classifier(training_set, training_labels)

            print 'predicting'
            predicted = classify(testing_set, classifier)
            error = error_measure(predicted, testing_labels)

            print 'error:', error
            results.append(error)
        print zip(training_set_sizes, results)
    else:
        results = [x[1] for x in training_set_sizes]
        training_set_sizes = [x[0] for x in training_set_sizes]
    plt.plot(training_set_sizes, results, 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('SVM Error vs Training set size')
    plt.xlabel('Training set size')
    plt.ylabel('Error')
    plt.show()


def grid_searchx():
    print 'loading data'
    training_set, training_labels, testing_set, testing_labels = select_data(5000, 100)
    # training_set, training_labels, testing_set, testing_labels = (
    #     pickle.load(open('training_set_2.p')),
    #     pickle.load(open('training_labels_2.p')),
    #     pickle.load(open('testing_set_2.p')),
    #     pickle.load(open('testing_labels_2.p')))
    # preprocessing
    print 'preprocessing data'
    training_set = preprocess(training_set, False)
    testing_set = preprocess(testing_set, False)
    parameters = {'kernel': ['rbf'], 'C': [8], 'gamma': [0, 0.02]}
    clf = grid_search.GridSearchCV(svm.SVC(), parameters)
    clf.fit(training_set, training_labels)
    print clf.best_params_, clf.best_score_


if __name__ == '__main__':
    generate_training_and_testing_sets()
    # problem_5b([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000], 1000)
    # problem_5b([(10, 0.566), (25, 0.525), (50, 0.357), (100, 0.321), (250, 0.263), (500, 0.193), (1000, 0.147),
    #             (2500, 0.107), (5000, 0.09), (10000, 0.072), (25000, 0.067), (50000, 0.06)])
    # print problem_5b2([(1, 'rbf', 3), (0.95, 'rbf', 3), (0.9, 'rbf', 3), (0.8, 'rbf', 3), (0.6, 'rbf', 3)])
    # plot_a_thing([((1, 'rbf', 3), 0.083), ((0.95, 'rbf', 3), 0.084), ((0.9, 'rbf', 3), 0.084),
    #               ((0.8, 'rbf', 3), 0.085), ((0.6, 'rbf', 3), 0.087)])
    # print problem_5b2([(1, 'rbf', 1), (1, 'rbf', 2), (1, 'rbf', 3), (1, 'rbf', 4), (1, 'rbf', 5)])
    # plot_a_thing([((1, 'rbf', 1), 0.083), ((1, 'rbf', 2), 0.083), ((1, 'rbf', 3), 0.083), ((1, 'rbf', 4), 0.083),
    #               ((1, 'rbf', 5), 0.083)])
    # print problem_5b2([(1, 'linear', 3), (1, 'poly', 3), (1, 'rbf', 3), (1, 'sigmoid', 3)])
    # plot_a_thing([((1, 'linear', 3), 0.091), ((1, 'poly', 3), 0.806), ((1, 'rbf', 3), 0.083), ((1, 'sigmoid', 3), 0.9)])
    # plot_a_thing([((0.1, 'rbf', 3), 0.144), ((0.5, 'rbf', 3), 0.092), ((1, 'rbf', 3), 0.083), ((5, 'rbf', 3), 0.071),
    #               ((10, 'rbf', 3), 0.059), ((50, 'rbf', 3), 0.055), ((100, 'rbf', 3), 0.053), ((500, 'rbf', 3), 0.048),
    #               ((1000, 'rbf', 3), 0.048), ((5000, 'rbf', 3), 0.048), ((10000, 'rbf', 3), 0.048),
    #               ((50000, 'rbf', 3), 0.048), ((100000, 'rbf', 3), 0.048)])
    # print problem_5b2([0.0001, 0.001, 0.01, 0.1, 1, 10])
    # plot_a_thing([(0.0001, 0.087), (0.001, 0.07), (0.01, 0.034), (0.1, 0.096), (1, 0.79), (10, 0.8)])
    # grid_searchx()
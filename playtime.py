import matplotlib.pyplot as plt
from mnist import load_mnist
import classifier_1 as c1
import pickle
import numpy as np


def problem_3b_sizes(training_set_sizes, testing_set_size=None):
    results = []
    for training_set_size in training_set_sizes:
        print "Starting", training_set_size

        print 'loading data'
        training_set, training_labels, testing_set, testing_labels = c1.select_data(training_set_size,
                                                                                    testing_set_size)
        testing_set = pickle.load(open('testing_set.p'))
        testing_labels = pickle.load(open('testing_labels.p'))

        print 'preprocessing data'
        training_set = c1.preprocess(training_set)
        testing_set = c1.preprocess(testing_set)

        print 'building classifier'
        classifier = c1.build_classifier(training_set, training_labels)

        print 'predicting'
        predicted = c1.classify(testing_set, classifier)
        error = c1.error_measure(predicted, testing_labels)

        print 'error:', error
        results.append(error)
    print zip(training_set_sizes, results)

    plt.plot(training_set_sizes, results, 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Naive Bayes Error vs Training set size')
    plt.xlabel('Training set size')
    plt.ylabel('Error')
    plt.show()


def problem_3b_parameters():
    results = []
    alphas = xrange(0, 100, 4)
    binarizes = np.array(xrange(0, 10))/10.0
    for binarize in binarizes:
        print "\nStarting Binarize:", binarize

        print 'loading data'
        training_set = pickle.load(open('training_set.p'))
        training_labels = pickle.load(open('training_labels.p'))
        testing_set = pickle.load(open('testing_set.p'))
        testing_labels = pickle.load(open('testing_labels.p'))

        print 'preprocessing data'
        training_set = c1.preprocess(training_set)
        testing_set = c1.preprocess(testing_set)

        print 'building classifier'
        classifier = c1.build_classifier(training_set, training_labels, binarize=binarize)

        print 'predicting'
        predicted = c1.classify(testing_set, classifier)
        error = c1.error_measure(predicted, testing_labels)

        print 'error:', error
        results.append(error)

    print zip(binarizes, results)
    plt.plot(binarizes, results, 'o')
    plt.title('Naive Bayes Error vs Threshold parameter')
    plt.xlabel('Threshold parameter')
    plt.ylabel('Error')
    plt.show()

    results = []

    for alpha in alphas:
        print "\nStarting Alpha:", alpha

        print 'loading data'
        training_set = pickle.load(open('training_set.p'))
        training_labels = pickle.load(open('training_labels.p'))
        testing_set = pickle.load(open('testing_set.p'))
        testing_labels = pickle.load(open('testing_labels.p'))

        print 'preprocessing data'
        training_set = c1.preprocess(training_set)
        testing_set = c1.preprocess(testing_set)

        print 'building classifier'
        classifier = c1.build_classifier(training_set, training_labels, alpha=alpha)

        print 'predicting'
        predicted = c1.classify(testing_set, classifier)
        error = c1.error_measure(predicted, testing_labels)

        print 'error:', error
        results.append(error)

    print zip(alphas, results)
    plt.plot(alphas, results, 'o')
    plt.title('Naive Bayes Error vs Alpha parameter')
    plt.xlabel('Alpha Value')
    plt.ylabel('Error')
    plt.show()


def main():
    # c1.select_data(10000, 1000, True)
    problem_3b_sizes([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000], 1000)
    # problem_3b_parameters()

if __name__ == '__main__':
    main()

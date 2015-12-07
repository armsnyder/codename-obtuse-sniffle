import pickle
from mnist import load_mnist
import numpy as np
import random
from sklearn import naive_bayes as nb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def preprocess(images):
    # this function is suggested to help build your classifier.
    # You might want to do something with the images before
    # handing them to the classifier. Right now it does nothing.
    return np.array([i.flatten() for i in images], dtype=float)


def build_classifier(images, labels, alpha=1.0, binarize=0.5, fit_prior=True):
    # this will actually build the classifier. In general, it
    # will call something from sklearn to build it, and it must
    # return the output of sklearn. Right now it does nothing.
    classifier = nb.BernoulliNB(alpha=alpha, binarize=binarize, fit_prior=fit_prior)
    classifier.fit(images, labels)
    return classifier


# the functions below are required
def save_classifier(classifier, training_set, training_labels):
    # this saves the classifier to a file "classifier" that we will
    # load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    pickle.dump(training_set, open('training_set.p', 'w'))
    pickle.dump(training_labels, open('training_labels.p', 'w'))


def classify(images, classifier):
    # runs the classifier on a set of images.
    return classifier.predict(images)


def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual)) / float(len(predicted))


def select_data(training_set_size, testing_set_size, save_test_set=False):
    """
    Choose examples for training and testing data, randomly, as described in 1C.
    :param training_set_size: Total number of training examples
    :param testing_set_size: Total number of testing examples
    :param save_test_set: If True, pickle testing set
    :return: training_set, training_labels, testing_set, testing_labels
    """
    training_set = []
    training_labels = []
    testing_set = []
    testing_labels = []

    for digit in xrange(10):
        # load digit:
        images, labels = load_mnist(digits=[digit], path='.')

        # choose random digits to add to training and testing sets:
        if (training_set_size+testing_set_size)/10 <= len(images):
            combined_sample_size = (training_set_size+testing_set_size)/10
            testing_sample_size = testing_set_size/10
        else:
            combined_sample_size = len(images)
            testing_sample_size = testing_set_size/10 * combined_sample_size / ((training_set_size+testing_set_size)/10)
        training_indices = random.sample(range(len(images)), combined_sample_size)
        testing_indices = random.sample(training_indices, testing_sample_size)
        training_indices = [x for x in training_indices if x not in testing_indices]

        # add to training set:
        training_set.extend(images[i] for i in training_indices)
        training_labels.extend(labels[i] for i in training_indices)

        # add to testing set:
        testing_set.extend([images[i] for i in testing_indices])
        testing_labels.extend([labels[i] for i in testing_indices])

    if save_test_set:
        pickle.dump(testing_set, open('testing_set.p', 'w'))
        pickle.dump(testing_labels, open('testing_labels.p', 'w'))
    return training_set, training_labels, testing_set, testing_labels


def main(load_classifier=False, load_data=False):
    # Code for loading data
    print 'loading data'
    if load_data:
        training_set = pickle.load(open('training_set.p'))
        training_labels = pickle.load(open('training_labels.p'))
        testing_set = pickle.load(open('testing_set.p'))
        testing_labels = pickle.load(open('testing_labels.p'))
    else:
        training_set, training_labels, testing_set, testing_labels = select_data(10000, 1000, True)

    # preprocessing
    print 'preprocessing data'
    training_set = preprocess(training_set)
    testing_set = preprocess(testing_set)

    # build_classifier is a function that takes in training data and outputs an sklearn classifier.
    if load_classifier:
        print 'loading classifier'
        classifier = pickle.load(open('classifier_1.p'))
    else:
        print 'building classifier'
        classifier = build_classifier(training_set, training_labels)
        print 'saving classifier'
        save_classifier(classifier, training_set, training_labels)

    print 'predicting'
    predicted = classify(testing_set, classifier)
    print error_measure(predicted, testing_labels)


def confusion():
    testing_set, testing_labels = (
            pickle.load(open('testing_set.p')),
            pickle.load(open('testing_labels.p')))
    testing_set = preprocess(testing_set)
    classifier = pickle.load(open('classifier_1.p'))
    predicted = classify(testing_set, classifier)
    print confusion_matrix(testing_labels, predicted)


def show_misclassified():
    testing_set, testing_labels = (
        pickle.load(open('testing_set.p')),
        pickle.load(open('testing_labels.p')))
    testing_set_p = preprocess(testing_set)
    classifier = pickle.load(open('classifier_1.p'))
    predicted = classify(testing_set_p, classifier)
    for i in range(len(predicted)):
        if predicted[i] != testing_labels[i]:
            plt.imshow(testing_set[i], cmap='gray')
            plt.title(str(testing_labels[i]) + ' misclassified as ' + str(predicted[i]))
            plt.show()


if __name__ == "__main__":
    show_misclassified()


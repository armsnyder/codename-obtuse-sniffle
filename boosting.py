from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
from sklearn import svm
import classifier_2


def boosting_A(training_set, training_labels, testing_set, testing_labels):
    # Build boosting algorithm for question A
    # Return confusion matrix
    classifier = AdaBoostClassifier()
    classifier.fit(training_set, training_labels)
    predicted_labels = classifier.predict(testing_set)
    print 'error:', error_measure(predicted_labels, testing_labels)
    return confusion_matrix(testing_labels, predicted_labels)


def boosting_B(training_set, training_labels, testing_set, testing_labels):
    # Build boosting algorithm for question B
    # Return confusion matrix
    print 'ready to boost'
    classifier = AdaBoostClassifier(base_estimator=svm.SVC(C=8, kernel='rbf', gamma=0.02), algorithm='SAMME')
    print 'fitting'
    classifier.fit(training_set, training_labels)
    print 'predicting'
    predicted_labels = classifier.predict(testing_set)
    print 'error:', error_measure(predicted_labels, testing_labels)
    return confusion_matrix(testing_labels, predicted_labels)


def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual)) / float(len(predicted))


def preprocess(images, binary=False):
    # this function is suggested to help build your classifier.
    # You might want to do something with the images before
    # handing them to the classifier. Right now it does nothing.
    if binary:
        return np.array([x > 0.5 for x in [i.flatten() for i in images]], dtype=bool)
    else:
        return np.array([i.flatten() for i in images], dtype=float)


def main():
    training_set, training_labels, testing_set, testing_labels = (
        pickle.load(open('training_set_2.p')),
        pickle.load(open('training_labels_2.p')),
        pickle.load(open('testing_set_2.p')),
        pickle.load(open('testing_labels_2.p'))
    ) if True else classifier_2.select_data(1000, 100)
    training_set = preprocess(training_set, False)
    testing_set = preprocess(testing_set, False)
    print boosting_B(training_set, training_labels, testing_set, testing_labels)


if __name__ == '__main__':
    main()
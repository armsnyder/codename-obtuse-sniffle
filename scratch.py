__author__ = 'flame'

import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
import random


def look_through_sevens(k):
    images, labels = load_mnist(digits=[7], path='.')
    # Displaying the mean image for digit 9.
    indices = random.sample(range(len(images)), k)
    for i in indices:
        plt.imshow(images[i], cmap='gray')
        plt.show()

if __name__ == '__main__':
    look_through_sevens(50)
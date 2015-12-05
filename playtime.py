import matplotlib.pyplot as plt
import numpy as np
import library
from mnist import load_mnist


def main():
    for i in xrange(7,8):
        images, labels = load_mnist(digits=[i], path='.')
        # for img in images:
        #     plt.imshow(img, cmap = 'gray')
        #     plt.show()
        plt.imshow(images.mean(axis=0), cmap='gray')
        plt.show()

if __name__ == '__main__':
    main()

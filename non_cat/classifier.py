import logging

logging.basicConfig(level=logging.INFO)

import sys
import scipy

from scipy     import ndimage
from train     import data
from PIL       import Image
from functions import predict

import numpy             as np
import matplotlib.pyplot as plt

def load_image(fname):
    return np.array(ndimage.imread(fname, flatten=False))

def classify(image):
    num_px = 64

    my_image = scipy.misc.imresize(image, size=(num_px, num_px))
    my_image = my_image.reshape((1, num_px*num_px*3)).T

    prediction = predict(data["w"], data["b"], my_image)

    return np.squeeze(prediction) == 1

if __name__ == "__main__":
    image = load_image(sys.argv[1])
    prediction = classify(image)

    if prediction:
        plt.title("cat detected 😸")
    else:
        plt.title("no cat detected 🙀")

    plt.imshow(image)
    plt.show()


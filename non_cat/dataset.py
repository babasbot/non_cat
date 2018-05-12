import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File("data/train_noncat.h5", "r")

    train_set_x_orig    = np.array(train_dataset["train_set_x"][:])
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    train_set_x         = train_set_x_flatten / 255.0

    train_set_y = np.array(train_dataset["train_set_y"][:])
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))

    test_dataset = h5py.File("data/test_noncat.h5", "r")

    test_set_x_orig    = np.array(test_dataset["test_set_x"][:])
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    test_set_x         = test_set_x_flatten / 255.0

    test_set_y = np.array(test_dataset["test_set_y"][:])
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    classes = np.array(test_dataset["list_classes"][:])

    return train_set_x, train_set_y, test_set_x, test_set_y, classes

train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()

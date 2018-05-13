import logging

logging.basicConfig(level=logging.INFO)

import dataset
import os.path
import h5py

import matplotlib.pyplot as plt
import numpy             as np

from model import model

logging.basicConfig(level=logging.INFO)

def train(iterations=2000, learning_rate=0.005, training_file="data/model.hdf5"):
    data = model(
        dataset.train_set_x,
        dataset.train_set_y,
        dataset.test_set_x,
        dataset.test_set_y,
        iterations=iterations,
        learning_rate=learning_rate
    )

    save_training(data, training_file)

    return data

def save_training(data, training_file="data/model.hdf5"):
    with h5py.File(training_file, 'a') as hf:
        for key in data:
            hf.create_dataset(key, data=np.array(data[key]))

def load_training(training_file="data/model.hdf5"):
    with h5py.File(training_file, 'r') as hf:
        data = {
            "costs":              hf["costs"][:],
            "Y_prediction_test":  hf["Y_prediction_test"][:],
            "Y_prediction_train": hf["Y_prediction_train"][:],
            "w":                  hf["w"][:],
            "b":                  hf["b"].value,
            "learning_rate":      hf["learning_rate"].value,
            "iterations":         hf["iterations"].value
        }

        return data

def plot_learning_curve():
    plt.plot(np.squeeze(data['costs']))
    plt.title("Learning rate = " + str(data["learning_rate"]))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

training_dataset = "data/model.hdf5"
if os.path.isfile(training_dataset):
    data = load_training(training_dataset)
else:
    data = train()

if __name__ == "__main__":
    plot_learning_curve()

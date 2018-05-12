import logging

logging.basicConfig(level=logging.INFO)

import numpy as np

import functions

def model(X_train, Y_train, X_test, Y_test, iterations=2500, learning_rate=0.005):
    w = np.zeros((X_train.shape[0], 1))
    b = 0

    parameters, grads, costs = functions.optimize(
        w,
        b,
        X_train,
        Y_train,
        iterations=iterations,
        learning_rate=learning_rate
    )

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test  = functions.predict(w, b, X_test)
    Y_prediction_train = functions.predict(w, b, X_train)

    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    logging.info(f"train accuracy: {train_accuracy} %")

    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    logging.info(f"test accuracy: {test_accuracy} %")

    return {
        "costs":              costs,
        "Y_prediction_test":  Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w":                  w,
        "b":                  b,
        "learning_rate":      learning_rate,
        "iterations":         iterations
    }

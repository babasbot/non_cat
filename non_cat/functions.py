import logging

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagate(w, b, X, Y):
    m = X.shape[1]

    # forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    # backward propagation
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    cost = np.squeeze(cost)

    grads = { "dw": dw, "db": db }

    return grads, cost

def optimize(w, b, X, Y, iterations, learning_rate):
    costs = []

    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)

        w = w - (learning_rate * grads['dw'])
        b = b - (learning_rate * grads['db'])

        if i % 100 == 0:
            logging.info(f"iteration: {i}, cost: {cost}")
            costs.append(cost)

        params = { "w": w, "b": b }

        grads = { "dw": grads["dw"], "db": grads["db"] }

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)

    Y_prediction = np.zeros((1,m))

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0, i] = 1

    return Y_prediction

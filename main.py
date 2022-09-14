import random
import numpy as np
from sklearn import datasets


def relu(t):
    """
    Activation function.
    :param t: array
    :return: return array with [[max(t1, 0) , max(t2,0) ...]]
    """
    return np.maximum(t, 0)


def softmax(t):
    """
    Standard function Softmax
    link: https://en.wikipedia.org/wiki/Softmax_function
    """
    ret = np.exp(t)
    return ret / np.sum(ret)


def sparse_cross_entropy(z, y):
    """y is not an array, so we just have to count -log(z[0,y]),
     instead of -sum(yi*log(z[0,yi]))"""
    return -np.log(z[0, y])


def to_full(y, num_classes):
    """
    :param y: index of 1
    :param num_classes: amount of classes
    :return: array of zeroes and one 1 by index of y
    """
    ret = np.zeros((1, num_classes))
    ret[0, y] = 1
    return ret


def relu_deriv(t):
    """Return 1 if t >= 0 and 0 if t < 0"""
    return (t >= 0).astype(float)


ALPHA = 0.001
EPOCHS = 500

INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 5

iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

W1 = np.random.randn(INPUT_DIM, H_DIM)
b1 = np.random.randn(1, H_DIM)
W2 = np.random.randn(H_DIM, OUT_DIM)
b2 = np.random.randn(1, OUT_DIM)

loss_arr = []

for ep in range(EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset)):

        x, y = dataset[i]

        # Forward
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax(t2)
        E = sparse_cross_entropy(z, y)

        # Backward
        y_full = to_full(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = dE_dt1

        # Update
        W1 = W1 - ALPHA * dE_dW1
        W2 = W2 - ALPHA * dE_dW2
        b1 = b1 - ALPHA * dE_db1
        b2 = b2 - ALPHA * dE_db2

        loss_arr.append(E)


def predict(x):
    """Make a prediction """
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


def calc_accuracy():
    """Calculate an accuracy"""
    correct = 0
    for (x, y) in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc

accuracy = calc_accuracy()
print("Accuracy: ", accuracy)

# draws a mistakes plot.
import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

c1 = np.array([[10, 10, 0]])
c2 = np.array([[20, 20, 1]])

DIM = 3
N = 500

u1 = np.concatenate((3.0 * np.random.randn(N, 2), np.zeros((N, 1))), axis=1)
u2 = np.concatenate((3.0 * np.random.randn(N, 2), np.zeros((N, 1))), axis=1)

train_data = np.concatenate((c1 + u1, c2 + u1), axis=0)
test_data = np.concatenate((c1 + u2, c2 + u2), axis=0)


def input_plot(g1, g2, title, color):
    plt.title(title)
    plt.scatter(g1[:, 0], g1[:, 1], s=10, color=color[0], alpha=0.5)
    plt.scatter(g2[:, 0], g2[:, 1], s=10, color=color[1], alpha=0.5)
    plt.show()


input_plot(train_data[:N, :], train_data[N:, :], title='training dataset', color=('blue', 'blue'))
input_plot(test_data[:N, :], test_data[N:, :], title='testing dataset', color=('red', 'red'))
input_plot(train_data, test_data, title="all dataset (overlapped)", color=('blue', 'red'))


def binary_classify(data):
    learning_rate = 0.5
    w = np.array([0, 0])  # u, v
    b = 0

    def init():
        return

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def distance(prob, ans):
        return -(ans * np.log(prob) + (1 - ans) * np.log(1 - prob))

    def loss():
        return 0

    def accuracy():
        return 0

    def dw(z):
        return (1 / N) * np.sum(data[:, 0:1] * (sigmoid(z) - data[:, 2]))

    def iterate():
        nonlocal w, b

        while True:
            w = w - learning_rate * (dw(np.dot(w.T, data) + b) - data[:, 2])

            break

    init()
    iterate()


binary_classify()

import matplotlib.pyplot as plt
import numpy as np

c1 = np.array([[10, 10, 0]])
c2 = np.array([[20, 20, 1]])

N = 500
TOTAL = N * 2

u1 = np.concatenate((3.0 * np.random.randn(N, 2), np.zeros((N, 1))), axis=1)
u2 = np.concatenate((3.0 * np.random.randn(N, 2), np.zeros((N, 1))), axis=1)

train_data = np.concatenate((c1 + u1, c2 + u1), axis=0).T
test_data = np.concatenate((c1 + u2, c2 + u2), axis=0).T


def input_plot(g1, g2, title, color):
    plt.title(title)
    plt.scatter(g1[0, :], g1[1, :], s=10, color=color[0], alpha=0.5)
    plt.scatter(g2[0, :], g2[1, :], s=10, color=color[1], alpha=0.5)
    plt.show()


def output_plot(g1, g2, title, color):
    plt.title(title)
    plt.plot(np.arange(1, len(g1)), g1, color=color[0])
    plt.plot(np.arange(1, len(g2)), g2, color=color[1])
    plt.show()


input_plot(train_data[:2, :N], train_data[:2, N:], title='training dataset', color=('blue', 'blue'))
input_plot(test_data[:2, :N], test_data[:2, N:], title='testing dataset', color=('red', 'red'))
input_plot(train_data[:2, :], test_data[:2, :], title="all dataset (overlapped)", color=('blue', 'red'))


def binary_classify(data):
    learning_rate = 0.015
    w = np.array([0, 0])  # u, v
    b = 0

    losses = []
    accuracies = []

    def init():
        return

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def distance(prob, ans):
        return -(np.nan_to_num(ans * np.log(prob)) + np.nan_to_num((1 - ans) * np.log(1 - prob)))

    def loss(prob, ans):
        return (1 / TOTAL) * np.nan_to_num(np.sum(distance(prob, ans)))

    def accuracy():
        return 0

    def dw(z):
        return (1 / TOTAL) * np.sum(data[:2, :] * (sigmoid(z) - data[2, :]), axis=1)

    def db(z):
        return (1 / TOTAL) * np.sum(sigmoid(z) - data[2, :])

    def iterate():
        p_loss = 0
        nonlocal w, b, losses, accuracies

        while True:
            z = np.dot(w.T, data[:2, :]) + b
            w = w - (learning_rate * dw(z))
            b = b - (learning_rate * db(z))

            n_loss = loss(sigmoid(z), data[2, :])
            losses.append(n_loss)

            if n_loss == p_loss:
                break
            else:
                print(n_loss)
                p_loss = n_loss
                continue

    init()
    iterate()

    return losses


train_loss = binary_classify(train_data)
test_loss = binary_classify(test_data)

output_plot(train_loss, test_loss, "loss", ('blue', 'red'))

import matplotlib.pyplot as plt
import numpy as np


c1 = np.array([[10, 10]])
c2 = np.array([[20, 20]])

N = 500
u1 = 3.0 * np.random.randn(N, 2)
u2 = 3.0 * np.random.randn(N, 2)

train_data = np.concatenate((c1+u1, c2+u1), axis=0)
test_data = np.concatenate((c1+u2, c2+u2), axis=0)


def input_plot(g1, g2, title, color):
    plt.title(title)
    plt.scatter(g1[:, 0], g1[:, 1], s=10, color=color[0], alpha=0.5)
    plt.scatter(g2[:, 0], g2[:, 1], s=10, color=color[1], alpha=0.5)
    plt.show()


input_plot(train_data[:N, :], train_data[N:, :], title='training dataset', color=('blue', 'blue'))
input_plot(test_data[:N, :], test_data[N:, :], title='testing dataset', color=('red', 'red'))
input_plot(train_data, test_data, title="all dataset (overlapped)", color=('blue', 'red'))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


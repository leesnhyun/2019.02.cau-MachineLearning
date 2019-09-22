import matplotlib.pyplot as plt
import numpy as np
import random

c1 = np.array([[10, 10]])
c2 = np.array([[20, 20]])

N = 500
u1 = 3.0 * np.random.randn(N, 2)
u2 = 3.0 * np.random.randn(N, 2)

def input_plot(g1, g2, title, color):
    plt.title(title)
    plt.scatter(g1[:, 0], g1[:, 1], s=10, color=color[0], alpha=0.5)
    plt.scatter(g2[:, 0], g2[:, 1], s=10, color=color[1], alpha=0.5)
    plt.show()


input_plot(c1+u1, c2+u1, title='training dataset', color=('blue', 'blue'))
input_plot(c1+u2, c2+u2, title='testing dataset', color=('red', 'red'))
input_plot(np.concatenate((c1+u1, c2+u1), axis=0), np.concatenate((c1+u2, c2+u2), axis=0),
           title="all dataset (overlapped)", color=('blue', 'red'))


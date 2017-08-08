import matplotlib.pyplot as plt
import numpy as np

# Basic functions
exp = lambda x: np.exp(x + 1e-6)  # any tensor
ln = lambda x: np.log(x + 1e-6)  # any tensor

# Other
linear = lambda X, W: np.dot(W.T, X)
d_linear_X = lambda X, W: W
d_linear_W = lambda X, W: X

# Activation function
sigmoid = lambda x: 1 / (1 + exp(-x))  # any tensor
d_sigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
relu = lambda x: np.maximum(0, x)
d_relu = lambda x: 0 if x < 0 else 1

# Layerwise activation
Softmax = lambda X: exp(X)/exp(X).sum() # x is a vector

# Distances
nll = lambda p, y: - y * ln(p) - (1 - y) * ln(1 - p)  # any tensor

# Gradients
d_nll = lambda p, y: (-y + p) / p * (1 - p)  # Derived on p
norm2 = lambda p, y: (p - y) ** 2
d_norm2 = lambda p, y: (p - y) / 2


# plot activations
def plot_activations(activation, range=[-5, 5, .1], show_id=False):
    x = np.arange(*range)
    y = map(activation, x)
    x_ = np.arange(min(y), max(y), 0.1)
    plt.plot(x, 0 * x)
    plt.plot(x, y)
    if show_id == True:
        plt.plot(x_, x_)
    plt.show()


def plot_activations_distributions(activation, n_subplots=4, xlim=[-2, 2], n_mean=0, n_range=1):
    y = (np.random.randn(100000) + n_mean) / n_range
    for i in range(n_subplots):
        plt.subplot(1, n_subplots, i+1)
        if i != 0:
            plt.xlim(xlim)
        plt.hist(y, 50)
        y = map(activation,  y)
        print "after %d iteration, mean is %.2f and std is %.2f" % (i, np.mean(y), np.std(y))
    plt.show()


 
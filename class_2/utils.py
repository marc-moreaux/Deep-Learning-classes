import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import keras
from keras.datasets import mnist


# Load MNIST
def load_MNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = x_train/255.
    x_test = x_test/255.

    # To categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


# Load a dataset of sighnal going upwards and downwards
def load_up_down(n_samples=60):

    def signal(a, b):
        # Returs f(x) = ax + b + epsilon
        x = np.array(range(5))
        y = a * x + b + np.random.rand(5) - .5
        return y

    # Generate up signals
    a = np.random.rand(n_samples) + 1
    b = np.random.rand(n_samples) - .5
    up = map(signal, a, b)

    # Generate down signals
    a = np.random.rand(n_samples) - 2
    b = np.random.rand(n_samples) - .5
    down = map(signal, a, b)

    # Build a dataset of them
    X = up + down
    Y = [0 for _ in range(n_samples)]
    Y += [1 for _ in range(n_samples)]

    # Randomize
    rand_idx = list(range(n_samples * 2))
    np.random.shuffle(rand_idx)
    X = np.array([X[i] for i in rand_idx])
    Y = np.array([Y[i] for i in rand_idx])

    # Assign train and test
    n_train_samples = n_samples * 2 * 80 / 100
    x_train, y_train = X[:n_train_samples], Y[:n_train_samples]
    x_test, y_test = X[n_train_samples:], Y[n_train_samples:]

    return (x_train, y_train), (x_test, y_test)


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


# plot activations (activations.py)
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



# Deep learning Framework !! Wouhou !!
class Trainer(object):
    """This is the main class to train a network"""
    def __init__(self, arg):
        super(Trainer, self).__init__()
        self.arg = arg


class Linear(object):
    """Class for a linear layer
    The input X for a Linear Layer is a vector"""
    def __init__(self, in_size, units, name=None, kernel_init=None, activation=None):
        super(Layer, self).__init__()
        self.arg = name
        self.units = units
        self.name = name
        self.activation = activation or (lambda x: x)
        self.activation = np.vectorize(self.activation)
        self.kernel_init = kernel_init or (lambda x: 0)
        self.kernel_init = np.vectorize(self.kernel_init)
        self.in_size = in_size
        
        # Define Weight vector 
        self.W = np.ndarray(n_in, units)
        self.W = self.kernel_init(self.W)

    def forward(self, X):
        Y = np.dot(self.W.T, X)
        return self.activation(Y)


    def backward(self, X):
        return X


class Trainer(object):
    """This class trains a simple linear regression

    params :
      X, Y : train set (X) and objective (Y)
      Ws : list with the Weights of the model
      alphas : learning rate of every weights

    Quickly, you also have to define some function :
    Pred: the prediction function p(X, Ws)
    Loss: The loss function L(Y, P)
    dWs: list of function to compute the gradients
    """
    def __init__(self, X, Y, Ws, alphas):
        super(Trainer, self).__init__()
        self.X = X
        self.Y = Y
        self.Ws = list(Ws)
        self.alphas = (alphas) * len(Ws) if type(alphas) == 'int' else alphas
        self.dWs = None
        self.pred = None
        self.loss = None
        self.losses = []

    def train(self, n_steps=100):
        for i, m_zip in enumerate(zip(self.Ws, self.dWs, self.alphas)):
            w, dw, alpha = m_zip
            self.Ws[i] = w - alpha * dw()

    def animated_train(self, is_notebook=False):
        # Draw initial plot
        # fig = plt.figure()
        # ax = plt.axes(xlim=(0, 23), ylim=(0, 1500))
        fig, axs = plt.subplots(3, 1)

        # Subplot1
        axs[0].set_xlim(0, 23)
        axs[0].set_ylim(0, 1500)
        # axs[0].scatter(self.X[0], self.Y, s=1)
        line, = axs[0].plot([], [], lw=1, c='r')

        # Subplot2 - 
        X_ = range(len(self.X[0]))
        axs[1].set_ylim(self.Y.min() * 0.9, self.Y.max() * 1.1)
        axs[1].plot(X_, self.Y, lw=1)
        line2, = axs[1].plot([], [], lw=1, c='r')

        # Subplot3 - loss
        X_ = range(500)
        axs[2].set_xlim(0, 100)
        axs[2].set_ylim(0, 1e0)
        line3, = axs[2].plot([], [], lw=1, c='r')

        # Initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            return line, line2, line3,

        # Animation function.
        def animate(i, *fargs):
            self = fargs[0]
            if i == 0:
                del self.losses[:]

            # Train
            self.train(50)

            # Subplot1
            # X_ = [range(0, 25), ]
            # Y_ = self.pred(X_)
            # line.set_data(X_[0], Y_)

            # # Subplot2 - Y render
            # X_ = [range(len(self.X[0])), ]
            # Y_ = self.pred(self.X)
            # line2.set_data(X_[0], Y_)

            # Subplot3 - loss
            self.losses.append(self.loss())
            X_ = range(i + 1)
            Y_ = self.losses
            line3.set_data(X_, Y_)

            return line, line2, line3,

        # Call the animator.
        anim = animation.FuncAnimation(fig, animate, frames=100,
                                       init_func=init, fargs=[self, ],
                                       interval=20, blit=True)
        if is_notebook is True:
            plt.close(anim._fig)
            return anim
        else:
            plt.show()


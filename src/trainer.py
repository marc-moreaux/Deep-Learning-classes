import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from activations import sigmoid
import utils


class Trainer(object):
    """This class applies standard gradient descent to a model to be defined
    by the user

    params :
      X, Y : train set (X) and objective (Y)
      Ws : list with the Weights of the model
      alphas : learning rate of every weights

    Quickly, you also have to define some function :
    Pred: the prediction function p(X, Ws)
    Loss: The loss function L(Y, P)
    dWs: list of function to compute the gradients
    """
    def __init__(self, X, Y, Ws, alphas, n_epochs=25):
        super(Trainer, self).__init__()
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Ws = list(Ws)
        self.alphas = (alphas) * len(Ws) if type(alphas) == 'int' else alphas
        self.f_dWs = None  # This is a function
        self.f_preds = None  # This is a function
        self.f_loss = None  # This is a function
        
        # For later purpose
        self.losses = []
        self.accuracies = []
        self.n_epochs = n_epochs

    def train(self, X, n_steps=100):
        for i, m_zip in enumerate(zip(self.Ws, self.f_dWs(X), self.alphas)):
            W, dW, alpha = m_zip
            self.Ws[i] = W - alpha * dW

    def animated_train(self, is_notebook=False):
        # Draw initial plot
        fig, axs = plt.subplots(2, 1)

        # Subplot1 - accuracy
        axs[0].set_xlim(0, self.n_epochs)
        axs[0].set_ylim(0, 1)
        line, = axs[0].plot([], [], lw=1, c='r')

        # Subplot2 - loss
        axs[1].set_xlim(0, self.n_epochs)
        axs[1].set_ylim(0, 10)
        line2, = axs[1].plot([], [], lw=1, c='r')

        # Initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            line2.set_data([], [])
            return line, line2,

        # Animation function.
        def animate(i, *fargs):
            self = fargs[0]
            if i == 0:  # Reset
                del self.losses[:]
                del self.accuracies[:]

            # Train
            X = self.X
            Y = self.Y
            self.train(X)

            # Log some results
            self.losses.append(self.f_loss(X))
            preds = self.f_preds(X)
            self.accuracies = (preds==Ys).mean()

            # Update subplots
            line.set_data(range(len(self.accuracies)), self.accuracies)
            line2.set_data(range(len(self.losses)), self.losses)

            return line, line2,

        # Call the animator.
        anim = animation.FuncAnimation(fig, animate,
                                       frames=100,
                                       init_func=init,
                                       fargs=[self, ],
                                       interval=20,
                                       blit=True)

        if is_notebook is True:
            plt.close(anim._fig)
            return anim
        else:
            plt.show()


(X, Y), (x_test, y_test) = utils.load_MNIST(True)
X = X.reshape((X.shape[0], -1))

# Initilize the parameters
Ws = [np.zeros(X.shape[1]), .0]  # One neuron with weight and bias
alphas = (0.0001, 0.01)  # update weigts of weight and bias


t = Trainer(X, Y, Ws, alphas)

# Define Prediction and Loss
def f_preds(X):
    linear = np.multiply(X, t.Ws[0]).sum(axis=-1) + t.Ws[1]
    sig = sigmoid(linear)
    return sig

def f_loss(X):
    loss = np.power((t.Y - t.f_preds(X)), 2) * 1 / 2.
    return loss.mean()

def f_dWs(X):
    dl_dp = t.Y - t.f_preds(X)
    dl_dsig = np.multiply(dl_dp, sigmoid(X) * (1 - sigmoid(X)))
    dl_dw0 = np.multiply(dl_dsig, X).mean()
    dl_dw1 = dl_dsig.mean()    
    return dl_dw0, dl_dw1

t.f_preds = f_preds
t.f_loss = f_loss
t.f_dWs = f_dWs
print(t.f_preds(X).shape)
print(t.f_loss(X))
t.animated_train()

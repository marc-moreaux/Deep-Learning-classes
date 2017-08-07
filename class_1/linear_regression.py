import quandl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
quandl.ApiConfig.api_key = "sdDHCTAB2MpTDqF3KcE7"

# For this tutorial, you shall install quandl:
#   $ sudo pip install quandl


def stock_arr_to_XY(stock_arr):
    x = map(lambda x: x[0].toordinal(), stock_arr)
    y = map(lambda y: y[1], stock_arr)
    return x, y


def filter_on_same_X(arr1, arr2):
    # x1, y1 = arr1
    # x2, y2 = arr2
    # if x1[i] is not in x2, remove (x1[i], y[i])
    x1, y1 = arr1
    x2, y2 = arr2

    rm_idx = []
    for i in range(len(x1)):
        if x1[i] not in x2:
            rm_idx.append(i)

    for i in rm_idx[::-1]:
        x1.pop(i)
        y1.pop(i)

    rm_idx = []
    for i in range(len(x2)):
        if x2[i] not in x1:
            rm_idx.append(i)

    for i in rm_idx[::-1]:
        x2.pop(i)
        y2.pop(i)

    return (x1, y1), (x2, y2)


def plot_data(arr1, arr2):
    # Plot the data
    x1, y1 = arr1
    x2, y2 = arr2

    # 2 subplots
    fig, axs = plt.subplots(2,1)
    
    # 1st subplot
    ax2 = axs[0].twinx()
    axs[0].plot(x1, y1, c='r')
    axs[0].set_ylabel('gold', color='r')
    axs[0].tick_params('y', colors='r')

    ax2.plot(x2, y2, c='b')
    ax2.set_ylabel('silver', color='b')
    ax2.tick_params('y', colors='b')

    # 2nd subplot
    axs[1].scatter(y1, y2, s=1)

    # Show
    fig.show()


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
        self.alphas = (alphas)*len(Ws) if type(alphas) == 'int' else alphas
        self.dWs = None
        self.pred = None
        self.loss = None
        
    def train(self, n_steps=100):
        for i, (w, dw, alpha) in enumerate(zip(self.Ws, self.dWs, self.alphas)):
            self.Ws[i] = w - alpha * dw()


    def animated_train(self, is_notebook=False):
        # Draw initial plot
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 23), ylim=(0, 1500))
        ax.scatter(self.X, self.Y, s=1)
        line, = ax.plot([], [], lw=1, c='r')

        # Initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return line,

        # Animation function.
        def animate(i, *fargs):
            self = fargs[0]
            self.train(50)

            X_ = range(0,25)
            Y_ = self.pred(X_)
            line.set_data(X_, Y_)
            return line,

        # Call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, 
                                       init_func=init, fargs=[self,],
                                       interval=20, blit=True)
        if is_notebook is True:
            plt.close(anim._fig)
            return anim
        else:
            plt.show()


if __name__ == '__main__':
    # We use the london market to get the stock values of gold and silver
    gold = quandl.get("LBMA/GOLD", returns="numpy", start_date="2015-01-01")
    silver = quandl.get("LBMA/SILVER", returns="numpy")

    # Retrieve gold value by date
    XY_gold = stock_arr_to_XY(gold)
    XY_silver = stock_arr_to_XY(silver)

    # Filter arrays to have gold and silver values of the same dates
    XY_gold, XY_silver = filter_on_same_X(XY_gold, XY_silver)
    x_gold, y_gold = XY_gold
    x_silver, y_silver = XY_silver

    # Plot the data
    plot_data(XY_silver, XY_gold)

    ###############################
    # Now comes the fun !!!
    _, X, Y = np.array(x_silver), np.array(y_silver), np.array(y_gold)

    Ws = [0.5, 0.5]
    alphas = (0.0001, 0.01)
    t = Trainer(X, Y, Ws, alphas)
    t.pred = lambda X : np.multiply(X, t.Ws[0]) + t.Ws[1]
    t.loss = lambda : np.power((t.Y - t.pred(X)), 2) * 1 / 2.
    dl_dp = lambda : -(t.Y - t.pred(X))
    dl_dw0 = lambda : np.multiply(dl_dp(), X).mean()
    dl_dw1 = lambda : dl_dp().mean()
    t.dWs = (dl_dw0, dl_dw1)
    t.animated_train()


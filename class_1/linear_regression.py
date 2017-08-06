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
    axs[1].scatter(y_silver, y_gold, s=1)

    # Show
    fig.show()


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
X, Y = np.array(x_silver), np.array(y_silver), np.array(y_gold)

# Lets hypthetize the gold is a linear function of the silver
# y = w*x + b

# Initialize parameters
w = np.random.rand(1)
b = 0

# Define prediction, error and parameters gradients
pred = lambda X: np.multiply(X, w) + b
err = lambda X, Y : np.abs(Y - pred(X))
dw = lambda X, Y: -(np.multiply((Y-pred(X)), X)).mean()
db = lambda X, Y: -(Y-pred(X)).mean()


def train(X, Y, n_steps):
    global w, b
    for _ in range(n_steps):
        w = w - 0.0001 * dw(X, Y)
        b = b - 0.001 * db(X, Y)


def animated_train():
    # Draw initial plot
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 23), ylim=(0, 1500))
    ax.scatter(X, Y, s=1)
    line, = ax.plot([], [], lw=1, c='r')

    # Initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # Animation function.
    def animate(params):
        train(X, Y, 200)

        X1 = range(0,25)
        line.set_data(X1, pred(X1))
        print w, b
        return line,

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   interval=10, blit=True)
    plt.show()


animated_train()
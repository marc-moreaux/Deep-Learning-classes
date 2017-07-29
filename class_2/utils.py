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
relu = lambda x: np.max(0, x)
d_relu = lambda x: 0 if x < 0 else 1

# Layerwise activation
Softmax = lambda X: exp(X)/exp(X).sum() # x is a vector

# Distances
nll = lambda p, y: - y * ln(p) - (1 - y) * ln(1 - p)  # any tensor
d_nll = lambda p, y: (-y + p) / p * (1 - p)  # Derived on p
norm2 = lambda p, y: (p - y) ** 2
d_norm2 = lambda p, y: (p - y) / 2
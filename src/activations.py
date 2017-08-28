"""In this files are implemented the activations and weights initialisations
"""
import numpy as np


# Basic functions
def exp(x):
    return np.exp(x + 1e-8)  # any tensor


def ln(x):
    return np.log(x + 1e-8)  # any tensor


# Activations
def sigmoid(x):
    return 1 / (1 + exp(-x))  # any tensor


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946    
    if x > 0:
        return scale * x
    return scale * (alpha * (np.exp(x) - 1))


# Layerwise activation
def softmax(X):
    e_x = exp(x)
    return e_x/e_x.sum() # x is a vector


# Distances
nll = lambda p, y: - y * ln(p) - (1 - y) * ln(1 - p)  # any tensor

import numpy as np

def reverse_sigmoid(y):
    return np.log(y / (1 - y + 1e-8))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dcp_perturb(y, gamma=1.0, beta=0.1, tau=0.9, nu=1000):
    y = np.clip(y, 1e-6, 1 - 1e-6)
    y_max = np.max(y)
    alpha = sigmoid(nu * (y_max - tau))
    perturb = sigmoid(gamma * reverse_sigmoid(y)) - 0.5
    y_perturbed = y - beta * alpha * perturb
    y_perturbed = np.clip(y_perturbed, 0, 1)
    return y_perturbed / np.sum(y_perturbed)


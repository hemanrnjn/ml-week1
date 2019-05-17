import numpy as np

def compute_cost_multi(X, y, theta):
    m = len(y)
    diff = X.dot(theta) - y
    J = 1.0 / (2 * m) * diff.T.dot(diff)
    return J
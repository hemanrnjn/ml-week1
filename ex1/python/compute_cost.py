import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    temp = 0
    for i in range(m):
        temp = temp + np.square(theta[0]*X[i, 0] + theta[1]*X[i, 1] - y[i])
    return temp / (2.0 * m)

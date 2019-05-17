import numpy as np
import compute_cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros(iterations)

    for i in range(iterations):
        theta -= ((X.dot(theta) - y).T.dot(X)) * alpha / m
        J_history[i] = compute_cost.compute_cost(X, y, theta)

    return theta, J_history
        
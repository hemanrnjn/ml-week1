import numpy as np
import matplotlib.pyplot as plt
import plot_data
import compute_cost
import gradient_descent

data = np.loadtxt(open("ex1data1.txt", "r"), delimiter=",")
X = data[:, 0]
y = data[:, 1]
m = len(y)

X = np.c_[np.ones(m), X]
theta = np.zeros(2)

iterations = 1500
alpha = 0.01

J = compute_cost.compute_cost(X, y, theta)
print(J)

theta, _ = gradient_descent.gradient_descent(X, y, theta, alpha, iterations)
print(theta)

plt.figure()
plot_data.plot_data(X[:, 1], y)
plt.plot(X[:, 1], X.dot(theta), label='Linear Regression')
plt.legend(loc='upper left', numpoints=1)
plt.show()


predict1 = np.array([1, 3.5]).dot(theta)
print("For population = 35,000, we predict a profit of", predict1 * 10000)


predict2 = np.array([1, 7]).dot(theta)
print("For population = 70,000, we predict a profit of", predict2 * 10000)
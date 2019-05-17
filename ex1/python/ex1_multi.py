import numpy as np
import normalize as nrm
import compute_cost_multi


data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

X, mu, sigma = nrm.normalize(X)
X = np.c_[np.ones(m), X]

iterations = 400
alpha = 0.15

theta = np.zeros(3)

J = compute_cost_multi.compute_cost_multi(X, y, theta)
print(J)

# theta, _ = gradient_descent.gradient_descent(X, y, theta, alpha, iterations)
# print(theta)

# plt.figure()
# plot_data.plot_data(X[:, 1], y)
# plt.plot(X[:, 1], X.dot(theta), label='Linear Regression')
# plt.legend(loc='upper left', numpoints=1)
# plt.show()


# predict1 = np.array([1, 3.5]).dot(theta)
# print("For population = 35,000, we predict a profit of", predict1 * 10000)


# predict2 = np.array([1, 7]).dot(theta)
# print("For population = 70,000, we predict a profit of", predict2 * 10000)
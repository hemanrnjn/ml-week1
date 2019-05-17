import numpy as np
import matplotlib.pyplot as plt

def plot_data(x, y):
    plt.plot(x, y, linestyle='', marker='x', color='r', label='Training data')
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')


data = np.loadtxt(open("ex1data1.txt", "r"), delimiter=",")
X = data[:, 0]
y = data[:, 1]
m = len(y)
plt.figure()
plot_data(X, y)
plt.show()
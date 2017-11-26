import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log(1.0 + np.exp(x))


fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)

x = np.linspace(-3, 3, 1000)
y = sigmoid(x)
tanh = 2 * sigmoid(2 * x) - 1
relu = np.where(x < 0, 0, x)
sp = softplus(x)

plt.xlim(-3, 3)
plt.ylim(-1.1, 2.1)

# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.set_xticks([-2, -1, 0, 1, 2])
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# ax.set_yticks([-1, 1, 2])

plt.axis([-3, 3, -1.1, 2.1])

width = 2.0

x2 = np.linspace(0, 10, 1000)
z = np.zeros(1000)

plt.plot(z, x, color='black')
plt.plot(x2, z, color='black')
plt.plot(x, y, label="Sigmoid", color="green", linewidth=width)
plt.plot(2 * x, tanh, label="Tanh", color="red", linewidth=width)
plt.plot(x, relu, label="ReLU", color='blue', linewidth=width)
plt.plot(x, sp, label="Softplus", color='brown', linewidth=width)
plt.legend(loc='upper left')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

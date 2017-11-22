import numpy as np
import matplotlib.pyplot as plt


def compare():
    x = np.arange(4)
    y = [0.7862, 0.8115, 0.7991, 0.8188]
    z = [0.8213, 0.9056, 0.8996, 0.9094]

    width = 0.3

    plt.bar(x+width, y, width=width)
    plt.bar(x+2*width, z, width=width)

    plt.legend()
    plt.show()

compare()

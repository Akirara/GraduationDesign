import numpy as np
import scipy as sp

x = np.array([1/3, 1/6, 1/4, 1/12, 1/10, 1/15])
y = np.array([1/4, 1/12, 1/3, 1/6, 1/15, 1/10])

cos = float(np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))


def kl(P, Q):
    return sum(P * sp.log(P / Q))

print(cos, kl(x, y))

import matplotlib.pyplot as plt
import numpy as np


plt.figure()

x = np.arange(6)
print(x.size)
y = [1.0/3, 1.0/6, 1.0/4, 1/12, 1/10, 1/15]

plt.bar(x, y)
plt.show()

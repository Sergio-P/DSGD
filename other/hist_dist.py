import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


x = np.random.normal(1, 1.5, size=(2000,))

mean = np.nanmean(x, axis=0)
std = np.nanstd(x, axis=0)
brks = norm.ppf(np.linspace(0, 1, 2+2))[1:-1]
print(mean + std * brks)

plt.hist(x, bins=100)
plt.plot([mean + std * brks[0], mean + std * brks[0]], [0, 75], 'k--')
plt.plot([mean + std * brks[1], mean + std * brks[1]], [0, 75], 'k--')

plt.ylim(0,75)
plt.show()
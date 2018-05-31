import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("result/perfomance.csv")

X_VAR = "k"
Y_VAR = "time"


if X_VAR != "N":
    df = df[(df.N == 1000)]
if X_VAR != "k":
    df = df[(df.k == 5)]
if X_VAR != "m":
    df = df[(df.m == 2)]

df = df.sort_values(by=[X_VAR])
x = df[X_VAR]
y = df[Y_VAR]

plt.plot(x,y,'o-')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x), "k--")

title = "%s = %.3f*%s + %.3f" % (Y_VAR, z[0], X_VAR, z[1])

plt.title(title)
plt.xlabel(X_VAR)
plt.ylabel(Y_VAR)

plt.show()

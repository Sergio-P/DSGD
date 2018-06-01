import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("result/exp_m.csv")

X_VAR = "m"
Y_VAR = "time"
split_plots = True


df = df.sort_values(by=[X_VAR])
x = df[X_VAR]
y = df[Y_VAR]

if not split_plots:
    plt.plot(x,df["time"],'o-')

    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x,p(x), "k--")

    title = "$%s = %.2f*%s^2 + %.2f*%s + %.2f$" % (Y_VAR, z[0], X_VAR, z[1], X_VAR, z[2])

    plt.title(title)

else:
    plt.plot(x,df["t_forward"],'o-',label="Forward Rule evaluation")
    plt.plot(x,df["t_loss"],'o-',label="Backward Loss computation")
    plt.plot(x,df["t_optim"],'o-',label="Updating masses")
    plt.plot(x,df["t_norm"],'o-',label="Normalizing masses")
    plt.legend()
    plt.title("Process time comparison")

plt.xlabel("Number of attributes")
plt.ylabel("Time [s]")

plt.show()

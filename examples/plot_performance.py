import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("result/exp_n.csv")
df2 = pd.read_csv("result/exp_n2.csv")

X_VAR = "N"
Y_VAR = "time"
split_plots = True


df = df.sort_values(by=[X_VAR])
x = df[X_VAR]
y = df[Y_VAR]

if not split_plots:
    plt.plot(x,df["time"],'o-')
    plt.plot(df2[X_VAR],df2["time"],'o-')

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x), "k--")

    # title = "%s = %.2f*%s^2 + %.2f*%s + %.2f" % (Y_VAR, z[0], X_VAR, z[1], X_VAR, z[2])
    title = "%s = %.4f*%s + %.2f" % (Y_VAR, z[0], X_VAR, z[1])
    print(title)

    z = np.polyfit(df2["N"], df2["time"], 1)
    p = np.poly1d(z)
    plt.plot(df2[X_VAR], p(x), "k--")

    plt.axis([0,3250,0,140])

    # title = "%s = %.2f*%s^2 + %.2f*%s + %.2f" % (Y_VAR, z[0], X_VAR, z[1], X_VAR, z[2])
    title = "%s = %.4f*%s + %.2f" % (Y_VAR, z[0], X_VAR, z[1])
    print(title)

    plt.title("Time")

else:
    plt.plot(x,df["t_forward"],'bo-',label="Forward Rule evaluation")
    plt.plot(x,df["t_loss"],'ro-',label="Backward Loss computation")
    plt.plot(x,df["t_optim"],'go-',label="Updating masses")
    plt.plot(x,df["t_norm"],'mo-',label="Normalizing masses")

    x = df2[X_VAR]
    plt.plot(x, df2["t_forward"], 'bo--', label="Forward Rule evaluation")
    plt.plot(x, df2["t_loss"], 'ro--', label="Backward Loss computation")
    plt.plot(x, df2["t_optim"], 'go--', label="Updating masses")
    plt.plot(x, df2["t_norm"], 'mo--', label="Normalizing masses")

    plt.axis([0, 3255, 0, 82])

    plt.legend()
    plt.title("Process time comparison")

plt.xlabel("Number of attributes")
plt.ylabel("Time [s]")

plt.show()

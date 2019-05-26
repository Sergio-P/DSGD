import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


df = pd.read_csv(sys.argv[1])

if len(df.columns) == 3:
    plt.scatter(df[df.columns[0]], df[df.columns[1]], c=df["cls"], cmap="bwr")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.show()
else:
    print("More than 3 columns found in dataset")



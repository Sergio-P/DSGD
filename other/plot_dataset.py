import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


df = pd.read_csv(sys.argv[1])

if len(df.columns) == 3:
    plt.scatter(df["attr0"], df["attr1"], c=df["cls"], cmap="RdBu")
    plt.show()
else:
    print("More than 3 columns found in dataset")



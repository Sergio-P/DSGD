import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


FILENAME_1 = "other/stroke-history-3-prc.csv"
NAME_1 = "DS-Stat"
FILENAME_2 = "other/stroke-19-prc.csv"
NAME_2 = "DS-GD"

# Random
plt.plot([0, 1], [0.2, 0.2], 'k--', label="Random")

# FILE 1
rcsv = pd.read_csv(FILENAME_1, header=None).values.reshape(-1)
n = int(len(rcsv)/2)
print("AUC score for DS %s: %.3f" % (NAME_1, auc(rcsv[n:], rcsv[:n])))
plt.plot(rcsv[n:], rcsv[:n], label=NAME_1)

# FILE 2
rcsv = pd.read_csv(FILENAME_2, header=None).values.reshape(-1)
n = int(len(rcsv)/2)
print("AUC score for DS %s: %.3f" % (NAME_2, auc(rcsv[n:], rcsv[:n])))
plt.plot(rcsv[n:], rcsv[:n], label=NAME_2)

plt.title("PR Curve for class Stroke")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.legend(loc="upper right")
plt.savefig("prc-comparison.png")

plt.show()
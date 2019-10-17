import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/stroke_data_18.csv")

# data["cls"].plot.hist()
#
# print(len(data))

# data = data.iloc[:, 9:-5]

# data.plot.scatter(x="LDL-C", y="HDL-C", c="cls", cmap="bwr", s=0.5)
# plt.xlim(0,250)
# plt.ylim(0,140)

# c = data.count(axis=0).sort_values(ascending=False)
# print(len(c))
# c.plot.bar(zorder=3, figsize=(10,5))

# BINS = [0,10,20,30,40,50,60,70,80,90,100,110]


# def bmi_index(r):
#     # print(r)
#     x = r.body_weights / (0.01 * r.heights) / (0.01 * r.heights)
#     if x <= 18.5:
#         return 0
#     elif x <= 25:
#         return 1
#     elif x <= 30:
#         return 2
#     elif x <= 100:
#         return 3
#     else:
#         return np.nan


# data["bmi"] = data.apply(bmi_index, axis=1)
#
datab = np.array([data[data.gender == 0].cls, data[data.gender == 1].cls])
print(datab[0].mean())
print(datab[1].mean())
#
#
plt.hist(datab, bins=[-.5,.5,1.5], histtype='bar', rwidth=0.9, zorder=3)
plt.xticks([0,1], ["No stroke", "Stroke"])
plt.legend(["Men", "Women"])

plt.grid(False)
plt.grid(axis="y", alpha=0.75, zorder=0)
plt.ylabel("Number of patients")
plt.title("Stroke prevalence")
plt.show()
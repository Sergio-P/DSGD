import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from ds.DSClassifierMulti import DSClassifierMulti

data = pd.read_csv("data/digits.csv", header=None)
data = data.rename(columns={64: "cls"})

data = data[(data.cls == 0) | (data.cls == 1)].reset_index(drop=True)  # Only 0s and 1s
data = data.sample(frac=1).reset_index(drop=True)

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
# X_train = (X_train >= 5).astype(int) + (X_train >= 10).astype(int)
X_train = (X_train >= 8).astype(int)
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
# X_test = (X_test >= 5).astype(int) + (X_test >= 10).astype(int)
X_test = (X_test >= 8).astype(int)
y_test = data.iloc[cut:, -1].values

# d0 = X_train[y_train == 0][:3].reshape((3,8,8))
# plt.subplot(231)
# plt.imshow(d0[0], cmap="Blues")
# plt.subplot(232)
# plt.imshow(d0[1], cmap="Blues")
# plt.subplot(233)
# plt.imshow(d0[2], cmap="Blues")
# d0 = X_train[y_train == 1][:3].reshape((3,8,8))
# plt.subplot(234)
# plt.imshow(d0[0], cmap="Blues")
# plt.subplot(235)
# plt.imshow(d0[1], cmap="Blues")
# plt.subplot(236)
# plt.imshow(d0[2], cmap="Blues")
#
# plt.show()
#
# exit()


DSC = DSClassifierMulti(2, max_iter=100, debug_mode=True, lr=0.0025)
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=1, print_every_epochs=1)
y_pred = DSC.predict(X_test)
print("Total Rules: %d" % DSC.model.get_rules_size())
print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
rls = DSC.model.find_most_important_rules(threshold=0.1)
dpx = {}
for cls in rls:
    rs = rls[cls]
    px = np.zeros(64)
    for r in rs:
        n = int(r[2].replace("X[", "").replace(" ", "").replace("]", "").split("=")[0])
        t = int(r[2].replace("X[", "").replace(" ", "").replace("]", "").split("=")[1])
        s = r[3][0]
        s = s if t == 1 else -s
        px[n] = s
    dpx[cls] = px

for cls in rls:
    px = dpx[1 - cls] - 0.5 * dpx[cls]
    plt.imshow(px.reshape((8, 8)), cmap="viridis_r")
    plt.colorbar()
    plt.title("Pixel contribution for class %d" % cls)
    plt.show()

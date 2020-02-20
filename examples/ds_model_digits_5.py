import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from dsgd.DSClassifierMulti import DSClassifierMulti

data = pd.read_csv("data/digits.csv", header=None)
data = data.rename(columns={64: "cls"})

data = data[data.cls <= 3].reset_index(drop=True)  # Only 0s and 1s
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

d0 = X_train[y_train == 0][:2].reshape((2,8,8))
plt.subplot(241)
plt.imshow(d0[0], cmap="Blues")
plt.subplot(242)
plt.imshow(d0[1], cmap="Blues")
d0 = X_train[y_train == 1][:2].reshape((2,8,8))
plt.subplot(243)
plt.imshow(d0[0], cmap="Blues")
plt.subplot(244)
plt.imshow(d0[1], cmap="Blues")
d0 = X_train[y_train == 2][:2].reshape((2,8,8))
plt.subplot(245)
plt.imshow(d0[0], cmap="Blues")
plt.subplot(246)
plt.imshow(d0[1], cmap="Blues")
d0 = X_train[y_train == 3][:2].reshape((2,8,8))
plt.subplot(247)
plt.imshow(d0[0], cmap="Blues")
plt.subplot(248)
plt.imshow(d0[1], cmap="Blues")

plt.show()

# exit()


DSC = DSClassifierMulti(4, max_iter=200, debug_mode=True)
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=1, print_every_epochs=1)
y_pred = DSC.predict(X_test)
print("Total Rules: %d" % DSC.model.get_rules_size())
print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(DSC.model.find_most_important_rules(class_names=["0", "1", "2", "3"]))
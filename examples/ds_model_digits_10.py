import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

from ds.DSClassifierMulti import DSClassifierMulti

data = pd.read_csv("data/digits.csv", header=None)
data = data.rename(columns={64: "cls"})

# data = data[data.cls <= 3].reset_index(drop=True)  # Only 0s and 1s
data = data.sample(frac=1).reset_index(drop=True)

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
X_train = (X_train >= 5).astype(int) + (X_train >= 10).astype(int)
# X_train = (X_train >= 8).astype(int)
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
X_test = (X_test >= 5).astype(int) + (X_test >= 10).astype(int)
# X_test = (X_test >= 8).astype(int)
y_test = data.iloc[cut:, -1].values

plt.figure(figsize=(14,6))

for i in range(10):
    d0 = X_train[y_train == i][:2].reshape((2,8,8))
    ax = plt.subplot(3, 7, 2*i + 1)
    plt.imshow(d0[0], cmap="Blues")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = plt.subplot(3, 7, 2*i + 2)
    plt.imshow(d0[1], cmap="Blues")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# exit()


DSC = DSClassifierMulti(10, max_iter=200, debug_mode=True, lossfn="MSE")
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=1, print_every_epochs=1)
y_pred = DSC.predict(X_test)
print("Total Rules: %d" % DSC.model.get_rules_size())
print("\nTraining Time: %.2f" % dt)
print("Epochs: %d" % (epoch + 1))
print("Min Loss: %.3f" % losses[-1])
print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# print(DSC.model.find_most_important_rules(class_names=["0", "1", "2", "3"]))
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from ds.DSClassifierGD import DSClassifier

data = pd.read_csv("data/digits.csv", header=None)
data = data.rename(columns={64: "cls"})

data = data[(data.cls == 0) | (data.cls == 1)].reset_index(drop=True)  # Only 0s and 1s
data = data.sample(frac=1).reset_index(drop=True)

# d0 = data[data.cls == 0].iloc[:3,:-1].values.reshape((3,8,8))
# plt.subplot(231)
# plt.imshow(d0[0], cmap="Blues")
# plt.subplot(232)
# plt.imshow(d0[1], cmap="Blues")
# plt.subplot(233)
# plt.imshow(d0[2], cmap="Blues")
# d0 = data[data.cls == 1].iloc[:3,:-1].values.reshape((3,8,8))
# plt.subplot(234)
# plt.imshow(d0[0], cmap="Blues")
# plt.subplot(235)
# plt.imshow(d0[1], cmap="Blues")
# plt.subplot(236)
# plt.imshow(d0[2], cmap="Blues")
#
# plt.show()

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].as_matrix()
y_train = data.iloc[:cut, -1].as_matrix()
X_test = data.iloc[cut:, :-1].as_matrix()
y_test = data.iloc[cut:, -1].as_matrix()

DSC = DSClassifier(max_iter=200, debug_mode=True)
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=1, print_every_epochs=1)
y_pred = DSC.predict(X_test)
print "Total Rules: %d" % DSC.model.get_rules_size()
print "\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.)
print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)

print DSC.model.find_most_important_rules(class_names=["0", "1"])
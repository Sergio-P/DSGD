import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

from ds.DSClassifierMulti import DSClassifierMulti
from ds.DSClassifierMultiQ import DSClassifierMultiQ

X_train = np.loadtxt("data/digits_kernels/x_train.csv", delimiter=",")
X_test = np.loadtxt("data/digits_kernels/x_test.csv", delimiter=",")
y_train = np.loadtxt("data/digits_kernels/y_train.csv", delimiter=",")
y_test = np.loadtxt("data/digits_kernels/y_test.csv", delimiter=",")

DSC = DSClassifierMultiQ(10, max_iter=100, lr=0.005, debug_mode=True, lossfn="MSE", precompute_rules=True)
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=2, print_every_epochs=1,
                            print_epoch_progress=True)
y_pred = DSC.predict(X_test)

# DSC = RandomForestClassifier(100)
# DSC.fit(X_train, y_train)
# y_pred = DSC.predict(X_test)

DSC.model.print_most_important_rules()
# print("Total Rules: %d" % DSC.model.get_rules_size())
# print("\nTraining Time: %.2f" % dt)
# print("Epochs: %d" % (epoch + 1))
# print("Min Loss: %.3f" % losses[-1])
print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

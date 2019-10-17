import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

from ds.DSClassifierMultiQ import DSClassifierMultiQ

X = np.random.rand(500, 1) * 10 - np.ones((500, 1)) * 5
y = (X[:, 0] > 0).astype(int)

n_class = len(np.unique(y))

DSC = DSClassifierMultiQ(n_class, max_iter=50, min_iter=50, step_debug_mode=True, min_dloss=0.0001, lr=0.025,
                         lossfn="MSE", precompute_rules=True)

DSC.model.add_rule(lambda x: x >= 0)
DSC.model.add_rule(lambda x: x < 0)

losses, epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm, masses = DSC.fit(X, y)
y_pred = DSC.predict(X)
# print(DSC.model.find_most_important_rules(class_names=["setosa", "virginica", "versicolor"]))
ac = accuracy_score(y, y_pred)
print("\nAccuracy:\t%.3f" % ac)
if n_class == 2:
    auc = roc_auc_score(y, y_pred)
    print("AUC ROC:\t%.3f" % auc)
else:
    auc = 0
f1mac = f1_score(y, y_pred, average="macro")
print("F1 Macro:\t%.3f" % f1mac)
f1mic = f1_score(y, y_pred, average="micro")
print("F1 Micro:\t%.3f" % f1mic)
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))


def pl(title, xl, yl):
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.grid(True)


it = np.arange(50)
plt.plot(it, losses)
pl("Training Loss", "Epochs", "MSE")
plt.show()

plt.plot(it, masses[:, 0, 0], label="Neg")
plt.plot(it, masses[:, 0, 1], label="Pos")
plt.plot(it, masses[:, 0, 2], label="Unc")
pl("Mass evolution for Rule A", "Epochs", "Values")
plt.legend()
plt.show()

plt.plot(it, masses[:, 1, 0], label="Neg")
plt.plot(it, masses[:, 1, 1], label="Pos")
plt.plot(it, masses[:, 1, 2], label="Unc")
pl("Mass evolution for Rule B", "Epochs", "Values")
plt.legend()
plt.show()

# print("%d,%.3f,%.3f,%.3f,%.3f,%.3f" % (epoch + 1, dt, ac, auc, f1mac, losses[-1]))

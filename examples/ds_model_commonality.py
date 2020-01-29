import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ds.DSClassifierMulti import DSClassifierMulti
from ds.DSClassifierMultiQ import DSClassifierMultiQ

# CONFIG
ms = []
qs = []
for i in range(5):
    X, y = make_blobs(500, n_features=2, random_state=20*i + 2, centers=3, cluster_std=2.5)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.3)

    n_class = len(np.unique(y))

    # MASS
    print("Mass Results")
    DSC = DSClassifierMulti(n_class, max_iter=100, min_iter=100, min_dloss=0.0005, lr=0.01, debug_mode=True,
                            lossfn="MSE", precompute_rules=True)

    losses, epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm = DSC.fit(X, y, add_single_rules=True,
                                                                        single_rules_breaks=7,
                                                                        print_partial_time=True,
                                                                        return_partial_dt=True)
    y_pred = DSC.predict(X_val)
    ac = accuracy_score(y_val, y_pred)
    print("\nAccuracy:\t%.3f" % ac)
    if n_class == 2:
        auc = roc_auc_score(y_val, y_pred)
        print("AUC ROC:\t%.3f" % auc)
    else:
        auc = 0
    f1mac = f1_score(y_val, y_pred, average="macro")
    print("F1 Macro:\t%.3f" % f1mac)
    f1mic = f1_score(y_val, y_pred, average="micro")
    print("F1 Micro:\t%.3f" % f1mic)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # COMMONALITY
    print("\n\nCommonality Results")
    DSC = DSClassifierMultiQ(n_class, max_iter=100, min_iter=100, debug_mode=True, min_dloss=0.0005, lr=0.01,
                             lossfn="MSE", precompute_rules=True)

    losses, epoch, dtq, dtq_forward, dtq_loss, dtq_optim, dtq_norm = DSC.fit(X, y, add_single_rules=True,
                                                                             single_rules_breaks=7,
                                                                             print_partial_time=True,
                                                                             return_partial_dt=True)
    y_pred = DSC.predict(X_val)
    ac = accuracy_score(y_val, y_pred)
    print("\nAccuracy:\t%.3f" % ac)
    if n_class == 2:
        auc = roc_auc_score(y_val, y_pred)
        print("AUC ROC:\t%.3f" % auc)
    else:
        auc = 0
    f1mac = f1_score(y_val, y_pred, average="macro")
    print("F1 Macro:\t%.3f" % f1mac)
    f1mic = f1_score(y_val, y_pred, average="micro")
    print("F1 Micro:\t%.3f" % f1mic)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    ms.append([dt, dt_forward, dt_loss, dt_optim, dt_norm])
    qs.append([dtq, dtq_forward, dtq_loss, dtq_optim, dtq_norm])

ms = np.array(ms)
qs = np.array(qs)

print(ms)
print(qs)

ind = -1 * np.arange(5)
width = 0.4
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(ind, ms.mean(axis=0), width, color='#1565C0', label='Masses', zorder=3, xerr=ms.std(axis=0), capsize=6)
ax.barh(ind - width, qs.mean(axis=0), width, color='#EF6C00', label='Commonality', zorder=3, xerr=qs.std(axis=0),
        capsize=6)

ax.set(yticks=ind - width/2, yticklabels=["Total", "Prediction", "Gradients", "Mass Updates", "Normalization"])
ax.legend(loc="lower right")
plt.grid(axis="x", zorder=0)
plt.xlabel("Time [s]")
plt.title("Commonality Transformation Comparison")

plt.show()

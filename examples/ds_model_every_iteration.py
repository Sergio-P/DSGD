import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ds.DSClassifierMultiQ import DSClassifierMultiQ
from ds.DSRule import DSRule

# CONFIG
# np.random.seed(123)
PLOT_MASSES = False
ALL_CLASSES_PLOT = True

X, y = make_blobs(500, n_features=2, random_state=42, centers=3, cluster_std=2.5)
# rng = np.random.RandomState(1)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)
#
# datasets = [make_moons(noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable
#             ]

# X, y = make_moons(500, noise=0.3, random_state=0)
# y = 1 - y

# X = np.random.rand(500, 2) * 2 - np.ones((500, 2))
# y = (X[:, 1] > 0).astype(int)

# data = pd.read_csv("data/f_ygauss.csv")
# data = data.sample(frac=1).reset_index(drop=True)
# print(data)
# X, X_val, y, y_val = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values, test_size=0.3)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.3)


plt.figure(figsize=(6, 6))
x_min, x_max = X[:, 0].min() * 1.2, X[:, 0].max() * 1.2
y_min, y_max = X[:, 1].min() * 1.2, X[:, 1].max() * 1.2
plt.axis((x_min, x_max, y_min, y_max))
plt.scatter(X[:, 0], X[:, 1], color=[cm.jet(float(yi) * .4 + .1) for yi in y])
plt.plot([x_min, x_max], [0, 0], 'k--')
plt.plot([0, 0], [y_min, y_max], 'k--')
plt.show()
# exit()

n_class = len(np.unique(y))


DSC = DSClassifierMultiQ(n_class, max_iter=150, min_iter=1, step_debug_mode=True, min_dloss=0.0005, lr=0.01,
                         lossfn="MSE", precompute_rules=True)

# DSC.model.add_rule(DSRule(lambda x: x[0] >= 0, "x >= 0"))
# DSC.model.add_rule(DSRule(lambda x: x[0] < 0, "x < 0"))
# DSC.model.add_rule(DSRule(lambda x: x[1] >= 0, "y >= 0"))
# DSC.model.add_rule(DSRule(lambda x: x[1] < 0, "y < 0"))

# DSC.model.generate_outside_range_pair_rules(pd.Series(["x", "y"]), [[-0.25, 0.5, "x"], [-0.5, 0.5, "y"]])

losses, epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm, masses = DSC.fit(X, y, add_single_rules=True,
                                                                            single_rules_breaks=6)
y_pred = DSC.predict(X_val)
# print(DSC.model.find_most_important_rules(class_names=["setosa", "virginica", "versicolor"]))
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


def pl(title, xl, yl):
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.grid(True)


it = np.arange(len(losses))
plt.plot(it, losses)
plt.ylim(0, .3)
pl("Training Loss", "Epochs", "MSE")
plt.show()


if PLOT_MASSES:
    for i, r in enumerate(DSC.model.preds):
        plt.plot(it, masses[:, i, 0], 'b', label="Blue")
        plt.plot(it, masses[:, i, 1], 'r', label="Red")
        plt.plot(it, masses[:, i, 2], 'g', label="Unc")
        pl("Mass evolution for Rule %s" % str(r), "Epochs", "Values")
        plt.legend()
        plt.show()

print(pd.DataFrame(masses[-1, :, :]).to_latex(float_format="%.3f"))

if X_val.shape[1] == 2:
    x_min, x_max = X_val[:, 0].min() * 1.2, X_val[:, 0].max() * 1.2
    y_min, y_max = X_val[:, 1].min() * 1.2, X_val[:, 1].max() * 1.2
    h = .5  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    N = len(yy.ravel())
    mesh = np.c_[xx.ravel(), yy.ravel()]

    if n_class == 2:
        Z = DSC.predict_proba(mesh)
        Z = Z[:, 1]
        # print(Z)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(8, 8))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)
        # Plot also the training points
        plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, edgecolors='k', cmap=plt.cm.coolwarm)
        plt.xlabel('x')
        plt.ylabel('y')

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        # plt.xticks(())
        # plt.yticks(())

        plt.show()
    else:
        if ALL_CLASSES_PLOT:
            Z = DSC.predict_proba(mesh)
            for i, c in enumerate(("Blues", "Greens", "Reds")):
                Zt = Z[:, i]
                # print(Z)

                # Put the result into a color plot
                Zt = Zt.reshape(xx.shape)
                plt.figure(1, figsize=(8, 8))
                plt.pcolormesh(xx, yy, Zt, cmap=c)

                # Plot also the training points
                plt.scatter(X_val[:, 0], X_val[:, 1], s=100, edgecolors='k', color=[cm.jet(float(yi) * .4 + .1) for yi in y_val])
                plt.xlabel('x')
                plt.ylabel('y')

                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                # plt.xticks(())
                # plt.yticks(())

                plt.show()

# print("%d,%.3f,%.3f,%.3f,%.3f,%.3f" % (epoch + 1, dt, ac, auc, f1mac, losses[-1]))

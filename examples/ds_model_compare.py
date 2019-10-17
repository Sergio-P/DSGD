import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from ds.DSClassifierMultiQ import DSClassifierMultiQ

DATSET = "SAheart"
print("DATASET: %s" % DATSET)

data = pd.read_csv("data/%s.csv" % DATSET)
data = data.sample(frac=1).reset_index(drop=True)

target, label_set = pd.factorize(data.iloc[:, -1])
n_class = len(label_set)
data.iloc[:, -1] = target

data = data.apply(pd.to_numeric, args=("coerce",))
data = data.dropna(axis=1)
# cut = int(0.65*len(data))

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


MODELS = ["DSCQ", "RF", "NB", "KNN", "MLP", "SVM"]

df_res = []

for model in MODELS:
    print("-"*45 + "\n" + model)
    if model == "DSCQ":
        cv = KFold(3)
        accs = []
        f1macs = []
        f1mics = []
        for train, test in cv.split(X, y):
            DSC = DSClassifierMultiQ(n_class, max_iter=300, min_iter=50, debug_mode=True, lr=0.005, min_dloss=0.0001,
                                     lossfn="MSE", precompute_rules=True, batch_size=500)
            losses, epoch, dt = DSC.fit(X[train], y[train], print_every_epochs=1, add_single_rules=True,
                                        single_rules_breaks=3)
            y_pred = DSC.predict(X[test])
            accs.append(accuracy_score(y[test], y_pred))
            # f1macs.append(f1_score(y[test], y_pred, average="macro"))
            # f1mics.append(f1_score(y[test], y_pred, average="micro"))
        df_res.append(["DSGD", *accs])
    else:
        if model == "RF":
            clf = RandomForestClassifier(100)
        elif model == "NB":
            clf = GaussianNB()
        elif model == "KNN":
            clf = KNeighborsClassifier(5)
        elif model == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, learning_rate_init=0.02)
        elif model == "SVM":
            clf = SVC(kernel="rbf", gamma="auto")
        res = cross_validate(clf, X, y, cv=3)
        df_res.append([model, *res["test_score"].tolist()])

    # f1mac = f1_score(y_test, y_pred, average="macro")
    # print("F1 Macro:\t%.3f" % f1mac)
    # f1mic = f1_score(y_test, y_pred, average="micro")
    # print("F1 Micro:\t%.3f" % f1mic)
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

df_res = pd.DataFrame(df_res, columns=["model", "Acc. Fold 1", "Acc. Fold 2", "Acc. Fold 3"])
df_res["Avg. Acc."] = (df_res["Acc. Fold 1"] + df_res["Acc. Fold 2"] + df_res["Acc. Fold 3"])/3.
print(df_res)

print(df_res.to_latex(index=False, float_format=lambda x: "%.3f" % x if type(x) != str else str(x),
                      column_format="rcccc"))
# print("%d,%.3f,%.3f,%.3f,%.3f,%.3f" % (epoch + 1, dt, ac, auc, f1mac, losses[-1]))

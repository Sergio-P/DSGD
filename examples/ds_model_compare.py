import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from ds.DSClassifierMultiQ import DSClassifierMultiQ

DATASETS = ["iris", "breast-cancer-wisconsin", "wine", "SAheart", "digits", "gas_drift"]

for DATASET in DATASETS:
    print("\n\n")
    print("-" * 60)
    print("DATASET: %s" % DATASET)

    data = pd.read_csv("data/%s.csv" % DATASET)
    data = data.sample(frac=1).reset_index(drop=True)

    target, label_set = pd.factorize(data.iloc[:, -1])
    n_class = len(label_set)
    print("Classes: %d" % n_class)
    data.iloc[:, -1] = target

    data = data.apply(pd.to_numeric, args=("coerce",))
    data = data.dropna(axis=1)
    # cut = int(0.65*len(data))

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    MODELS = ["DSCQ", "RF", "NB", "KNN", "MLP", "SVM"]

    df_res = []

    for model in MODELS:
        print("%s running..." % model)
        if model == "DSCQ":
            cv = KFold(3)
            accs = []
            f1macs = []
            aucs = []
            for train, test in cv.split(X, y):
                DSC = DSClassifierMultiQ(n_class, max_iter=1000, min_iter=1, debug_mode=False, lr=0.001, min_dloss=0.0001,
                                         lossfn="MSE", precompute_rules=True, batch_size=600)
                losses, epoch = DSC.fit(X[train], y[train], add_single_rules=True,
                                        print_epoch_progress=True, print_every_epochs=1,
                                        add_mult_rules=False, single_rules_breaks=3)
                print("Rules: %d" % DSC.model.get_rules_size())
                print("Epochs: %d" % epoch)
                print("Min Tr Loss: %.4f" % losses[-1])
                y_score = DSC.predict_proba(X[test])
                _, y_pred = torch.max(torch.Tensor(y_score), 1)
                y_pred = y_pred.numpy()
                # print(accuracy_score(y[test], y_pred))
                print(accuracy_score(y[test], y_pred))
                accs.append(accuracy_score(y[test], y_pred))
                f1macs.append(f1_score(y[test], y_pred, average="macro"))
                if n_class > 2:
                    aucs.append(roc_auc_score(y[test], y_score, multi_class="ovr"))
                else:
                    aucs.append(roc_auc_score(y[test], y_score[:, 1]))

            df_res.append(["DSGD", *accs, *f1macs, *aucs])
        else:
            if model == "RF":
                clf = RandomForestClassifier(50)
            elif model == "NB":
                clf = GaussianNB()
            elif model == "KNN":
                clf = KNeighborsClassifier(5)
            elif model == "MLP":
                clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, learning_rate_init=0.02)
            elif model == "SVM":
                clf = SVC(kernel="rbf", gamma="auto", probability=True, max_iter=10000)
            elif model == "Majority":
                clf = DummyClassifier(strategy="most_frequent")

            if n_class > 2:
                res = cross_validate(clf, X, y, cv=3, scoring=["accuracy", "f1_macro", "roc_auc_ovr"])
                df_res.append([model, *res["test_accuracy"].tolist(), *res["test_f1_macro"].tolist(),
                               *res["test_roc_auc_ovr"].tolist()])
            else:
                res = cross_validate(clf, X, y, cv=3, scoring=["accuracy", "f1_macro", "roc_auc"])
                df_res.append([model, *res["test_accuracy"].tolist(), *res["test_f1_macro"].tolist(),
                               *res["test_roc_auc"].tolist()])

        # f1mac = f1_score(y_test, y_pred, average="macro")
        # print("F1 Macro:\t%.3f" % f1mac)
        # f1mic = f1_score(y_test, y_pred, average="micro")
        # print("F1 Micro:\t%.3f" % f1mic)
        # print("\nConfusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))

    cols = ["model"]
    cols.extend(["acc%d" % x for x in range(1, 4)])
    cols.extend(["f1%d" % x for x in range(1, 4)])
    cols.extend(["auc%d" % x for x in range(1, 4)])

    df_res = pd.DataFrame(df_res, columns=cols)
    df_res["accuracy"] = (df_res["acc1"] + df_res["acc2"] + df_res["acc3"]) / 3.
    df_res["f1_macro"] = (df_res["f11"] + df_res["f12"] + df_res["f13"]) / 3.
    df_res["roc_auc"] = (df_res["auc1"] + df_res["auc2"] + df_res["auc3"]) / 3.
    print(df_res.to_string())

    print("\n")
    print(df_res.to_latex(index=False, float_format=lambda x: "%.3f" % x if type(x) != str else str(x),
                          column_format="rcccc", columns=["model", "accuracy", "f1_macro", "roc_auc"]))

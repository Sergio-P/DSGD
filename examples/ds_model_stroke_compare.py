import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import torch
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer as Imputer
from sklearn.svm import SVC

from dsgd.DSClassifierMultiQ import DSClassifierMultiQ

data = pd.read_csv("data/stroke_data_18.csv")

# data = data.drop("0", axis=1)
data = data.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=10)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

cut = int(0.7 * len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

print("Majority:")
print(1.*len(y_test[y_test == 0])/len(y_test))

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.fit_transform(X_test)

MODELS = ["DSGD", "RF", "NB", "KNN", "MLP", "SVM"]

df_res = []

plt.figure()
plt.plot([0, 1], [0, 1], 'k--', label="Random")

for model in MODELS:
    print("-"*45 + "\n" + model)
    if model == "DSGD":
        DSC = DSClassifierMultiQ(2, min_iter=50, max_iter=100, debug_mode=True, num_workers=4, lossfn="MSE",
                                 optim="adam", precompute_rules=True, batch_size=200, lr=0.005)
        DSC.model.load_rules_bin("stroke2.dsb")

        y_score = DSC.predict_proba(X_test)
        _, y_pred = torch.max(torch.Tensor(y_score), 1)
        y_pred = y_pred.numpy()
        y_score = y_score[:, 1]
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
        clf.fit(X_train_imp, y_train)
        y_score = clf.predict_proba(X_test_imp)
        y_score = y_score[:, 1]
        y_pred = clf.predict(X_test_imp)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_score)

    print("Accuracy: %.4f" % acc)
    print("ROC AUC: %.4f" % roc)

    df_res.append([model, acc, roc])

    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, label=model)


df_res = pd.DataFrame(df_res, columns=["Model", "Accuracy", "AUC ROC"])
print(df_res)

print(df_res.to_latex(index=False, float_format=lambda x: "%.3f" % x if type(x) != str else str(x),
                      column_format="rcc"))

plt.legend(loc="lower right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve for Stroke Prediction")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.show()


# print("%d,%.3f,%.3f,%.3f,%.3f,%.3f" % (epoch + 1, dt, ac, auc, f1mac, losses[-1]))

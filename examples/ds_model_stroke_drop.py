import matplotlib.pyplot as plt
import torch

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from dsgd.DSClassifierMultiQ import DSClassifierMultiQ

data = pd.read_csv("data/stroke_data_18.csv")

data = data.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=10)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=1).reset_index(drop=True)


X_test = data.iloc[:, :-1].values
y_test = data.iloc[:, -1].values


plt.figure()
plt.plot([0, 1], [0, 1], 'k--', label="Random")

res = []

for n in [200, 100, 75, 64, 50, 36, 24, 16, 12, 8, 4]:
    DSC = DSClassifierMultiQ(2, min_iter=50, max_iter=100, debug_mode=True, num_workers=4, lossfn="MSE", optim="adam",
                             precompute_rules=True, batch_size=200, lr=0.005)
    DSC.model.load_rules_bin("stroke2.dsb")
    DSC.model.keep_top_rules(n, imbalance=[1.224, 0.816])
    num_rules = DSC.model.get_active_rules_size()
    print("\n\nRules: %d" % num_rules)

    y_score = DSC.predict_proba(X_test)
    _, y_pred = torch.max(torch.Tensor(y_score), 1)
    y_pred = y_pred.numpy()

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: %.1f%%" % (acc * 100.))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_score[:, 1])
    print("AUC score: %.3f" % roc_auc)

    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    plt.plot(fpr, tpr, label="%d Rules" % num_rules)

    res.append([num_rules, acc, f1, roc_auc])


df = pd.DataFrame(res, columns=["model", "accuracy", "f1_macro", "roc_auc"])
print(df.to_latex(index=False, float_format=lambda x: "%.3f" % x if type(x) != str else str(x),
                  column_format="rccc"))

plt.legend(loc="lower right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve for Stroke Prediction")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.show()

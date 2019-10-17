import json
import numpy as np
import lime
import lime.lime_tabular
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Imputer

from ds.DSClassifierMulti import DSClassifierMulti
from ds.DSRule import DSRule

data_raw = pd.read_csv("data/stroke_data_18.csv")

# data = data.drop("0", axis=1)
data = data_raw.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=1)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=1).reset_index(drop=True)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
data = pd.DataFrame(imp.fit_transform(data), columns=data.columns)


cut = int(0.7 * len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

RF = RandomForestClassifier(100)

RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
y_score = RF.predict_proba(X_test)

print("Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("AUC score: %.3f" % (roc_auc_score(y_test, y_score[:, 1])))

# Plotting #
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ROC Curve
plt.figure()
plt.plot([0, 1], [0, 1], 'k--', label="Random")
fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
# np.savetxt("%s-roc.csv" % EXP_PREFIX, np.concatenate((fpr, tpr)))
plt.plot(fpr[1:-1], tpr[1:-1], label="DS Model")
plt.legend(loc="lower right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve for Stroke Prediction")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.show()
# plt.savefig("%s-roc.png" % EXP_PREFIX)

# PR Curve
plt.figure()
pre, rec, _ = precision_recall_curve(y_test, y_score[:, 1])
# np.savetxt("%s-prc.csv" % EXP_PREFIX, np.concatenate((pre, rec)))
plt.xlabel("Recall")
plt.plot(rec[1:-1], pre[1:-1])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve for Stroke prediction")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.show()
# plt.savefig("%s-prc.png" % EXP_PREFIX)


explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=data.columns,
                                                   class_names=["No stroke", "Stroke"], discretize_continuous=True)

while True:
    i = np.random.randint(0, X_test[y_test == 1].shape[0])
    if RF.predict(X_test[y_test == 1][i].reshape(1, -1)) == 1:
        xt = X_test[y_test == 1][i]
        print(xt)
        exp = explainer.explain_instance(xt, RF.predict_proba, num_features=20, top_labels=1)
        k = data.apply(lambda x: (np.abs(x[-1] - xt) < 0.01).all(), axis=1)
        print(k.unique())
        i = k[k][0].index
        print(data_raw[i])
        exp.save_to_file("exaplanation.html")
    # plt.show()

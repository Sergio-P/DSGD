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

from dsgd.DSClassifierMulti import DSClassifierMulti
from dsgd.DSClassifierMultiQ import DSClassifierMultiQ
from dsgd.DSRule import DSRule

data_raw = pd.read_csv("data/stroke_data_18.csv")

# data = data.drop("0", axis=1)
data = data_raw.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=1)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=1).reset_index(drop=True)

# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# data = pd.DataFrame(imp.fit_transform(data), columns=data.columns)


cut = int(0.7 * len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

# RF = RandomForestClassifier(100)
#
# RF.fit(X_train, y_train)

DSC = DSClassifierMultiQ(2, min_iter=50, max_iter=100, debug_mode=True, num_workers=4, lossfn="MSE", optim="adam",
                         precompute_rules=True, batch_size=200, lr=0.005)
DSC.model.load_rules_bin("stroke2.dsb")
y_pred = DSC.predict(X_test)
y_score = DSC.predict_proba(X_test)

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


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train_imp = imp.fit_transform(X_train)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_imp, feature_names=data.columns,
                                                   class_names=["No stroke", "Stroke"], discretize_continuous=True)

c = 0
while c < 4:
    i = np.random.randint(0, X_test[y_test == 0].shape[0])
    if DSC.predict(X_test[y_test == 0][i].reshape(1, -1)) == 1:
        c += 1
        xt = X_test[y_test == 0][i]
        print(xt)
        exp = explainer.explain_instance(xt, DSC.predict_proba, num_features=10, top_labels=1)
        # k = data.apply(lambda x: (np.abs(x[-1] - xt) < 0.01).all(), axis=1)
        # print(k.unique())
        # i = k[k][0].index
        exp = exp.as_list(label=1)
        fig = plt.figure(figsize=(6, 3))
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors, zorder=3)
        plt.yticks(pos, names)
        # exp.as_pyplot_figure(label=0)
        plt.title("No Stroke instance #%d classified as Stroke" % c)
        plt.grid(True, zorder=0)
        plt.show()
        # exp.save_to_file("exaplanation.html")
    # plt.show()

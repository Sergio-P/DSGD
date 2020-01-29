import json
import torch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from ds.DSClassifierMultiQ import DSClassifierMultiQ
from ds.DSRule import DSRule

data = pd.read_csv("data/stroke_data_18.csv")

# data = data.drop("0", axis=1)
data = data.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=10)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=1).reset_index(drop=True)


X_test = data.iloc[:, :-1].values
y_test = data.iloc[:, -1].values


DSC = DSClassifierMultiQ(2, min_iter=50, max_iter=100, debug_mode=True, num_workers=4, lossfn="MSE", optim="adam",
                         precompute_rules=True, batch_size=200, lr=0.005)
DSC.model.load_rules_bin("stroke2.dsb")

y_score = DSC.predict_proba(X_test)
_, y_pred = torch.max(torch.Tensor(y_score), 1)
y_pred = y_pred.numpy()

print("Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("AUC score: %.3f" % (roc_auc_score(y_test, y_score[:, 1])))
# DSC.model.print_most_important_rules(["No Stroke", "Stroke"], 0.15)
#
# # Plotting #
# import json
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
EXP_PREFIX = "stroke-19"
#
#
# with open("rules.json", "w") as f:
#     rules = DSC.model.find_most_important_rules(["No Stroke", "Stroke"], 0.2)
#     json.dump(rules, f)
#
#
# # Loss curve
# plt.figure()
# plt.plot(range(len(losses)), np.array(losses))
# plt.xlabel("Epochs")
# plt.ylabel("MSE")
# plt.title("Training Error")
# plt.savefig("%s-error.png" % EXP_PREFIX)
#
# # ROC Curve
plt.figure()
plt.plot([0, 1], [0, 1], 'k--', label="Random")
fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
# np.savetxt("%s-roc.csv" % EXP_PREFIX, np.concatenate((fpr,tpr)))
plt.plot(fpr[1:-1], tpr[1:-1], label="DS Model")
plt.legend(loc="lower right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve for Stroke Prediction")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.show()
# plt.savefig("%s-roc.png" % EXP_PREFIX)
#
# # PR Curve
plt.figure()
pre, rec, _ = precision_recall_curve(y_test, y_score[:, 1])
# np.savetxt("%s-prc.csv" % EXP_PREFIX, np.concatenate((pre,rec)))
plt.xlabel("Recall")
plt.plot([0, 1], [0.192, 0.192], 'k--', label="Random")
plt.plot(rec[1:-1], pre[1:-1], label="DS Model")
plt.xlabel("Recall")
plt.legend(loc="upper right")
plt.ylabel("Precision")
plt.title("PR Curve for Stroke prediction")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.show()
# plt.savefig("%s-prc.png" % EXP_PREFIX)

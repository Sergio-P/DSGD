import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from ds.DSClassifierGD import DSClassifier

data = pd.read_csv("data/stroke_data.csv")

data = data.drop("0", axis=1)
data = data.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=10)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=1).reset_index(drop=True)

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values


DSC = DSClassifier(max_iter=200, debug_mode=True, balance_class_data=True, num_workers=4)
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=5, add_mult_rules=False,
                            column_names=data.columns[:-1], print_every_epochs=1)
y_pred = DSC.predict(X_test)
y_score = DSC.predict_proba(X_test)

print "\nTraining Time: %.1f" % dt
print "Epochs: %d" % epoch
print "Total Rules: %d" % len(DSC.model.get_rules_size())
print "Min Loss: %.3f" % losses[-1]
print "Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.)
print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)
print "AUC score: %.3f" % (roc_auc_score(y_test, y_score))

print DSC.model.find_most_important_rules(threshold=0.2, class_names=["No Stroke", "Stroke"])

# Plotting #
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_PREFIX = "stroke-1"

# Loss curve
plt.plot(range(len(losses)), np.array(losses)/4)
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy")
plt.title("Training Error")
plt.savefig("%s-error.png")

# ROC Curve
plt.plot([0, 1], [0, 1], 'k--', label="Random")
fpr, tpr, _ = roc_curve(y_test, y_score)
np.savetxt("%s-roc.csv" % EXP_PREFIX, np.concatenate((fpr,tpr)))
plt.plot(fpr[1:-1], tpr[1:-1], "DS Model")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve for Stroke Prediction")
plt.grid("on")
plt.axis([0, 1, 0, 1])
plt.savefig("%s-roc.png" % EXP_PREFIX)

pre, rec, _ = precision_recall_curve(y_test, y_score)
np.savetxt("%s-prc.csv" % EXP_PREFIX, np.concatenate((pre,rec)))
plt.xlabel("Recall")
plt.plot(rec[1:-1], pre[1:-1])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve for Stroke prediction")
plt.grid("on")
plt.axis([0, 1, 0, 1])
plt.savefig("%s-prc.png" % EXP_PREFIX)

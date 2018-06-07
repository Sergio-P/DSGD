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

# Add custom rules
DSC.model.generate_categorical_rules(X_train, data.columns[:-1], exclude=["gender"])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "WBC", [4000, 10000])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "Ht", [38])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "MCV", [88, 102])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "MCH", [27, 32])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "MCHC", [30, 35])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "PLT", [10, 40])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "UA", [2, 7])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "LDL-C", [120, 160])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "CREA", [0.5, 1.2])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "TG", [100, 200])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "HDL-C", [40, 60])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "GOT", [30, 100])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "GPT", [30, 100])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "ALP", [200, 1400])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "GGT", [30, 100])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "blood_glucose", [100, 126, 200])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "HbA1C/JDS", [6, 7, 9, 11])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "bmis", [18, 25, 30, 35, 40])
DSC.model.generate_custom_range_single_rules(data.columns[:-1], "age", [80])
DSC.model.generate_custom_range_rules_by_gender(data.columns[:-1], "Hb", [7, 10, 13], [7, 10, 12])
DSC.model.generate_custom_range_rules_by_gender(data.columns[:-1], "body_fat", [8, 20], [14, 24])
DSC.model.generate_custom_range_rules_by_gender(data.columns[:-1], "waist_measurements", [102], [88])

losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=False, add_mult_rules=False, print_every_epochs=1)
y_pred = DSC.predict(X_test)
y_score = DSC.predict_proba(X_test)

print "\nTraining Time: %.1f" % dt
print "Epochs: %d" % epoch
print "Total Rules: %d" % DSC.model.get_rules_size()
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
plt.figure()
plt.plot(range(len(losses)), np.array(losses)/4)
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy")
plt.title("Training Error")
plt.savefig("%s-error.png" % EXP_PREFIX)

# ROC Curve
plt.figure()
plt.plot([0, 1], [0, 1], 'k--', label="Random")
fpr, tpr, _ = roc_curve(y_test, y_score)
np.savetxt("%s-roc.csv" % EXP_PREFIX, np.concatenate((fpr,tpr)))
plt.plot(fpr[1:-1], tpr[1:-1], label="DS Model")
plt.legend(loc="lower right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve for Stroke Prediction")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.savefig("%s-roc.png" % EXP_PREFIX)

# PR Curve
plt.figure()
pre, rec, _ = precision_recall_curve(y_test, y_score)
np.savetxt("%s-prc.csv" % EXP_PREFIX, np.concatenate((pre,rec)))
plt.xlabel("Recall")
plt.plot(rec[1:-1], pre[1:-1])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve for Stroke prediction")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.savefig("%s-prc.png" % EXP_PREFIX)

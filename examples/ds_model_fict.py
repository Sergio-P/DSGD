import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

from ds.DSClassifierMulti import DSClassifierMulti

data = pd.read_csv(sys.argv[1])
data = data.sample(frac=1).reset_index(drop=True)

TRAIN_TEST_FRAC = 0.7

data = data.apply(pd.to_numeric)
cut = int(TRAIN_TEST_FRAC*len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

n_class = len(np.unique(y_train))

DSC = DSClassifierMulti(n_class, max_iter=500, debug_mode=True, min_dloss=0.0001, lr=0.002, lossfn="MSE")

losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=3, add_mult_rules=True,
                            column_names=data.columns[:-1], print_every_epochs=1, print_final_model=True)
y_pred = DSC.predict(X_test)
# print(DSC.model.find_most_important_rules(class_names=["setosa", "virginica", "versicolor"]))
print("\nAccuracy:\t%.3f%%" % (accuracy_score(y_test, y_pred)))
# print("AUC ROC:\t%.3f%%" % (roc_auc_score(y_test, y_pred)))
print("F1 Macro:\t%.3f" % (f1_score(y_test, y_pred, average="macro")))
print("F1 Micro:\t%.3f" % (f1_score(y_test, y_pred, average="micro")))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
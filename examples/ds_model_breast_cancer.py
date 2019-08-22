import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score

from ds.DSClassifierMulti import DSClassifierMulti
from ds.DSClassifierMultiQ import DSClassifierMultiQ

data = pd.read_csv("data/breast-cancer-wisconsin.csv")

data = data.drop("id", axis=1)
data["class"] = data["class"].map({2: 0, 4: 1})

data = data.apply(pd.to_numeric, args=("coerce",))
data = data.sample(frac=1).reset_index(drop=True)

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values


DSC = DSClassifierMultiQ(2, max_iter=100, debug_mode=True, lossfn="MSE", min_dloss=0.0001, lr=0.01,
                         precompute_rules=True)
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=3,
                            column_names=data.columns[:-1], print_every_epochs=1)
y_pred = DSC.predict(X_test)
y_score = DSC.predict_proba(X_test)

print("\nTraining Time: %.1f" % dt)
print("Epochs: %d" % (epoch + 1))
print("Min Loss: %.4f" % losses[-1])
print("Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))
print("F1 Micro: %.3f" % (f1_score(y_test, y_pred, average="micro")))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("AUC score: %.3f" % (roc_auc_score(y_test, y_score[:, 1])))

DSC.model.print_most_important_rules(threshold=0.2, classes=["Benign", "Malignant"])

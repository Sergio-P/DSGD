import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score

from ds.DSClassifierGD import DSClassifier

data = pd.read_csv("data/SAheart.csv")

data = data.drop("row.names", axis=1)
data["famhist"] = data["famhist"].map({"Absent": 0, "Present": 1})

data = data.apply(pd.to_numeric, args=("coerce",))
data = data.sample(frac=1).reset_index(drop=True)

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values


DSC = DSClassifier(max_iter=400, debug_mode=True, lr=0.001, min_dloss=0.0001, balance_class_data=False)
losses, epoch, dt = DSC.fit(X_train, y_train, column_names=data.columns[:-1], add_single_rules=True,
                            single_rules_breaks=2, print_every_epochs=5, print_final_model=True)
y_pred = DSC.predict(X_test)
y_score = DSC.predict_proba(X_test)

print("\nTraining Time: %.1f" % dt)
print("Epochs: %d" % epoch)
print("Min Loss: %.3f" % losses[-1])
print("Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))
print("F1 Micro: %.3f" % (f1_score(y_test, y_pred, average="micro")))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("AUC score: %.3f" % (roc_auc_score(y_test, y_score)))

print(DSC.model.find_most_important_rules(threshold=0.24, class_names=["Healthy", "Heart Disease"]))
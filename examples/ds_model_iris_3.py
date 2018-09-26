import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from ds.DSClassifierMulti import DSClassifierMulti

data = pd.read_csv("data/iris.csv")
data = data.sample(frac=1).reset_index(drop=True)

data = data.replace("setosa", 0).replace("virginica", 1).replace("versicolor", 2)

data = data.apply(pd.to_numeric)
cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

DSC = DSClassifierMulti(3, max_iter=500, debug_mode=True, min_dloss=0.0001, lr=0.002, lossfn="MSE")
# DSC.model.load_rules_bin("rules.bin")
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=5, add_mult_rules=True,
                            column_names=data.columns[:-1], print_every_epochs=10, print_final_model=True)
y_pred = DSC.predict(X_test)
# print(DSC.model.find_most_important_rules(class_names=["setosa", "virginica", "versicolor"]))
print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# DSC.model.save_rules_bin("rules.bin")
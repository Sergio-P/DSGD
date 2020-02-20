import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from dsgd.DSClassifierGD import DSClassifier

data = pd.read_csv("data/iris.csv")
data = data[data.species != "versicolor"].reset_index(drop=True)  # Remove virginica to make the problem binary
data = data.sample(frac=1).reset_index(drop=True)

data = data.replace("setosa", 0).replace("virginica", 1)

data = data.apply(pd.to_numeric)
cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

DSC = DSClassifier(max_iter=200)
# DSC.model.load_rules_bin("rules.bin")
losses, epoch = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=5, add_mult_rules=True,
                        column_names=data.columns[:-1])
y_pred = DSC.predict(X_test)
print(DSC.model.find_most_important_rules(class_names=["setosa", "virginica"]))
print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# DSC.model.save_rules_bin("rules.bin")
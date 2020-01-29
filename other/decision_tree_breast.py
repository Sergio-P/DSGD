import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

DATASET = "breast-cancer-wisconsin"

data = pd.read_csv("data/%s.csv" % DATASET)
data = data.sample(frac=1).reset_index(drop=True)
data = data.apply(pd.to_numeric, args=("coerce",))
data = data.dropna(axis=1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

cut = int(0.7 * len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

clf = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5, criterion="entropy").fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy: %.4f" % acc)

plt.figure(figsize=(20,20))
plot_tree(clf, filled=True, feature_names=data.columns[:-1])
plt.show()
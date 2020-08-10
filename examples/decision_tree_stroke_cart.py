import pandas as pd
from dtreeviz.trees import dtreeviz
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer as Imputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv("data/stroke_data_18.csv")

# data = data.drop("0", axis=1)
data = data.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=10)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

cut = int(0.7 * len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

print("Majority:")
print(1.*len(y_test[y_test == 0])/len(y_test))

imp = Imputer()
X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.fit_transform(X_test)

X = imp.fit_transform(X)

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.05, class_weight={0: 1, 1: 1.2}, min_impurity_split=0.05)
tree.fit(X_train_imp, y_train)

y_pred = tree.predict(X_test_imp)
y_score = tree.predict_proba(X_test_imp)[:, 1]

# print("\nEpochs: %d" % epoch)
# print("Min Loss: %.4f" % ls)
print("Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("AUC score: %.3f" % (roc_auc_score(y_test, y_score)))
# print(DSC.model.find_most_important_rules(threshold=0.2, class_names=["No Stroke", "Stroke"]))

# print("\nAccuracy: %.4f" % accuracy)

plt.figure(figsize=(18,12))
plot_tree(tree, feature_names=data.columns[:-1], class_names=["No Stroke", "Stroke"], filled=True, proportion=True,
          rounded=True)
plt.savefig("tree.svg")

exit()

viz = dtreeviz(tree, X, y, target_name='stroke', feature_names=data.columns[:-1],
               class_names=["No Stroke", "Stroke"], histtype='barstacked', fancy=False,
               colors={"classes": [None, None, ["#81C784", "#E57373"]]})

viz.view()

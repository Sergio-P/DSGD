import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from ds.DSClassifierMulti import DSClassifierMulti

data = pd.read_csv("data/iris.csv")
data = data.sample(frac=1).reset_index(drop=True)

data = data.replace("setosa", 0).replace("virginica", 1).replace("versicolor", 2)

data = data.apply(pd.to_numeric)
# cut = int(0.7*len(data))

X = data.iloc[:, [0,2]].values
y = data.iloc[:, -1].values
# X_test = data.iloc[cut:, :-1].values
# y_test = data.iloc[cut:, -1].values

DSC = DSClassifierMulti(3, max_iter=200, debug_mode=True, min_dloss=0.0001, lr=0.002, lossfn="MSE", skip_dr_norm=False)
# DSC.model.load_rules_bin("rules.bin")
losses, epoch, dt = DSC.fit(X, y, add_single_rules=True, single_rules_breaks=3, add_mult_rules=True,
                            column_names=data.columns[:-1], print_every_epochs=3, print_final_model=False)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .05  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
N = len(yy.ravel())
mesh = np.c_[xx.ravel(), yy.ravel()]
Z = DSC.predict(mesh)
print(Z)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
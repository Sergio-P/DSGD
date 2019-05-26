import numpy as np
import pandas as pd
import sys
from sklearn.datasets import make_blobs


# X = np.random.rand(500, 2) * 2 - np.ones((500, 2))
# y = (X[:, 1] > 0).astype(int)

N = 500
m = 2
k = 2

X, y = make_blobs(n_samples=N, n_features=m, centers=2, cluster_std=0.25, random_state=16, center_box=(-1,1))
# c1 = (1.15*X[:, 0] - X[:, 1]).reshape(-1, 1)
# c2 = (-0.5*X[:, 0] + 0.75*X[:, 1]).reshape(-1, 1)
# X = np.concatenate((X, np.random.rand(N, 2), c1, c2), axis=1)
# y = ((y == 2) | (y == 1)).astype(int)
# X = np.concatenate((X, (X[:, 1] > 0).astype(int).reshape(-1, 1)), axis=1)

# df = pd.DataFrame(X, columns=["attr%d" % x for x in range(X.shape[1])])
df = pd.DataFrame(X, columns=["X", "Y"])
df["cls"] = y

df.to_csv(sys.argv[1], index=False)

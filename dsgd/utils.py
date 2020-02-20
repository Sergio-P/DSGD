import numpy as np
from sklearn.cluster import KMeans


def natural_breaks(data, k=5, append_infinity=False):
    km = KMeans(n_clusters=k, max_iter=150, n_init=5)
    data = map(lambda x: [x], data)
    km.fit(data)
    breaks = []
    if append_infinity:
        breaks.append(float("-inf"))
        breaks.append(float("inf"))
    for i in range(k):
        breaks.append(max(map(lambda x: x[0], filter(lambda x: km.predict(x) == i, data))))
    breaks.sort()
    return breaks


def statistic_breaks(data, k=5, sigma_tol=1, append_infinity=False):
    mu = np.mean(data)
    sigma = np.std(data)
    lsp = np.linspace(mu - sigma * sigma_tol, mu + sigma * sigma_tol, k)
    if append_infinity:
        return [float("-inf")] + [x for x in lsp] + [float("inf")]
    else:
        return [x for x in lsp]


def is_categorical(arr, max_cat = 6):
    # if len(df.unique()) <= 10:
    #     print df.unique()
    return len(np.unique(arr[~np.isnan(arr)])) <= max_cat


def normalize(a, b, c):
    a = 0 if a < 0 else 1 if a > 1 else a
    b = 0 if b < 0 else 1 if b > 1 else b
    c = 0 if c < 0 else 1 if c > 1 else c
    n = float(a + b + c)
    return a/n, b/n, c/n


def one_hot(n, k):
    a = np.zeros((len(n), k))
    for i in range(len(n)):
        a[i, int(n[i])] = 1
    return a


def h_center(z):
    return np.exp(- z * z)


def h_right(z):
    return (1 + np.tanh(z - 1))/2


def h_left(z):
    return (1 - np.tanh(z + 1))/2
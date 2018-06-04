# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, confusion_matrix

from ds.DSClassifierGD import DSClassifier

N = 1000
k = 3
m = 2

print "N,k,m,Ac,mloss,epochs,time,t_forward,t_loss,t_optim,t_norm"

if True:
# for N in np.linspace(50, 3050, 10):
    N = int(N)

    X, y = make_blobs(n_samples=N, n_features=m, centers=4, cluster_std=1., random_state=10)

    y = y/2

    cut = int(0.7 * len(X))

    X_train = X[:cut, :]
    y_train = y[:cut]
    X_test = X[cut:, :]
    y_test = y[cut:]

    DSC = DSClassifier(max_iter=200, lr=0.01, debug_mode=True, use_softmax=True, skip_dr_norm=True)
    losses, epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm = DSC.fit(X_train, y_train, add_single_rules=True,
                                                                        single_rules_breaks=k, add_mult_rules=True,
                                                                        return_partial_dt=True, disable_all_print=True)
    y_pred = DSC.predict(X_test)

    # print DSC.model.find_most_important_rules()

    accuracy = accuracy_score(y_test, y_pred)
    # print "Accuracy in test: %.1f%%" % (accuracy * 100)
    # print "Confusion Matrix"
    # print confusion_matrix(y_test, y_pred)

    print "%d,%d,%d,%f,%f,%d,%f,%f,%f,%f,%f" % (
    N, k, m, accuracy, losses[-1], epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm)

    print DSC.model.find_most_important_rules()

    plt.subplot(121)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)

    plt.subplot(122)
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iterations")
    plt.ylabel("CE")
    # plt.axis([0, len(losses) - 1, losses[-1] - 0.05, losses[0] + 0.02])
    plt.title("Error")
    plt.show()

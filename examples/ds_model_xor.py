import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from ds.DSModel import DSModel
from ds.DSRule import DSRule

X = np.random.rand(400, 2) * 2 - np.ones((400, 2))
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

cut = int(0.7*len(X))

X_train = X[:cut, :]
y_train = y[:cut]
X_test = X[cut:, :]
y_test = y[cut:]

accs = []
lbls = []
lss = []

for use_proper_rules in [False, True]:
    print("\n" + "-" * 40 + "\n")
    model = DSModel()
    if use_proper_rules:
        model.add_rule(DSRule(lambda x: x[0] * x[1] > 0, "Positive multiplication"))
        model.add_rule(DSRule(lambda x: x[0] * x[1] <= 0, "Negative multiplication"))
        # model.generate_statistic_single_rules(X_train, breaks=4)
    else:
        model.generate_statistic_single_rules(X_train, breaks=3)
        model.generate_mult_pair_rules(X_train)

    optimizer = torch.optim.Adam(model.masses, lr=.01)
    criterion = MSELoss()

    losses = []

    print(model)

    ti = time.time()
    model.train()
    epoch = 0
    Xt = Variable(torch.Tensor(X_train))
    # yt = Variable(torch.Tensor(y_train).long())
    yt = torch.Tensor(y_train).view(len(y_train), 1)
    yt = Variable(torch.cat([yt == 0, yt == 1], 1).float())

    for epoch in range(1000):
        # print "Processing epoch %d" % (epoch + 1)
        y_pred = model.forward(Xt)
        loss = criterion(y_pred, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize()
        losses.append(loss.data.item())
        if epoch > 2 and abs(losses[-2] - loss.data.item()) < 0.001:
            break

    print("Training time: %.2fs, epochs: %d" % (time.time() - ti, epoch))
    print("Least training loss reached: %.3f" % losses[-1])
    model.eval()

    # TESTING
    with torch.no_grad():
        print(model)
        print(model.find_most_important_rules())
        Xt = torch.Tensor(X_test)
        Yt = torch.Tensor(y_test).long().numpy()
        _, yt_pred = torch.max(model(Xt), 1)
        yt_pred = yt_pred.numpy()
        lbls.append(yt_pred)
        accuracy = accuracy_score(Yt, yt_pred)
        accs.append(accuracy)
        lss.append(losses)
        print("Accuracy in test: %.1f%%" % (accuracy * 100))
        print("Confusion Matrix")
        print(confusion_matrix(Yt, yt_pred))


plt.figure(figsize=(7,6))

plt.subplot(221)
plt.plot(range(len(lss[0])), lss[0], label="M1")
plt.plot(range(len(lss[1])), lss[1], label="M2")
plt.xlabel("Iterations")
plt.ylabel("CE")
plt.legend()
plt.title("Error")


plt.subplot(222)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="RdYlBu")
plt.title("Real labels")


plt.subplot(223)
plt.scatter(X_test[:,0], X_test[:,1], c=lbls[0], cmap="RdYlBu")
plt.title("Model 1 Ac: %d%%" % int(accs[0]*100))

plt.subplot(224)
plt.scatter(X_test[:,0], X_test[:,1], c=lbls[1], cmap="RdYlBu")
plt.title("Model 2 Ac: %d%%" % int(accs[1]*100))

plt.suptitle('XOR Classification using DS', fontsize=16)
plt.subplots_adjust(wspace=0.3, hspace=0.4)

plt.show()
# coding=utf-8

import time

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from ds.DSModel import DSModel

N = 1000
k = 3
m = 2

X, y = make_blobs(n_samples=N, n_features=m, centers=4, cluster_std=1.5, random_state=42)

y = y%2

cut = int(0.7*len(X))

X_train = X[:cut, :]
y_train = y[:cut]
X_test = X[cut:, :]
y_test = y[cut:]

model = DSModel()
model.generate_statistic_single_rules(X_train, breaks=k)
model.generate_mult_pair_rules(X_train)

optimizer = torch.optim.Adam(model.masses, lr=.01)
criterion = CrossEntropyLoss()

losses = []

# print model

dt_forward = 0
dt_loss = 0
dt_optim = 0
dt_norm = 0
ti = time.time()

model.train()
epoch = 0
Xt = Variable(torch.Tensor(X_train))
# yt = Variable(torch.Tensor(y_train).long())
yt = torch.Tensor(y_train).view(len(y_train), 1)
yt = Variable(torch.cat([yt == 0, yt == 1], 1).float())

for epoch in range(1000):
    print "Processing epoch %d" % (epoch + 1)

    tq = time.time()
    y_pred = model.forward(Xt)
    dt_forward += time.time() - tq

    tq = time.time()
    loss = criterion(y_pred, yt)
    optimizer.zero_grad()
    loss.backward()
    dt_loss += time.time() - tq

    tq = time.time()
    optimizer.step()
    dt_optim += time.time() - tq

    tq = time.time()
    model.normalize()
    dt_norm += time.time() - tq

    losses.append(loss.data.item())
    if epoch > 2 and abs(losses[-2] - loss.data.item()) < 0.001:
        break

dt = time.time() - ti
print "\nTraining time: %.2fs, epochs: %d" % (dt, epoch)

print "├- Forward eval time:  %.3fs" % dt_forward
print "├- Loss backward time: %.3fs" % dt_loss
print "├- Optimization time:  %.3fs" % dt_optim
print "└- Normalization time: %.3fs" % dt_norm

print "\nLeast training loss reached: %.3f" % losses[-1]
model.eval()
# print model

# TESTING
with torch.no_grad():
    print model.find_most_important_rules()
    Xt = torch.Tensor(X_test)
    Yt = torch.Tensor(y_test).long().numpy()
    _, yt_pred = torch.max(model(Xt), 1)
    yt_pred = yt_pred.numpy()
    accuracy = accuracy_score(Yt, yt_pred)
    print "Accuracy in test: %.1f%%" % (accuracy * 100)
    print "Confusion Matrix"
    print confusion_matrix(Yt, yt_pred)


plt.subplot(121)
plt.scatter(X_test[:,0], X_test[:,1], c=yt_pred)
# plt.show()

plt.subplot(122)
plt.plot(range(len(losses)), losses)
plt.xlabel("Iterations")
plt.ylabel("CE")
# plt.axis([0, len(losses) - 1, losses[-1] - 0.05, losses[0] + 0.02])
plt.title("Error")
plt.show()

print "N,k,m,Ac,time,epochs"
print "%d,%d,%d,%.3f,%.2f,%d" % (N, k, m, accuracy, dt, epoch)

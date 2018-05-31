import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from ds.DSModel import DSModel

N = 1000
k = 9
m = 2

X, y = make_blobs(n_samples=N, n_features=m, centers=2, cluster_std=1.5, random_state=42)

cut = int(0.7*len(X))

X_train = X[:cut, :]
y_train = y[:cut]
X_test = X[cut:, :]
y_test = y[cut:]

model = DSModel()
model.generate_statistic_rules(X_train, breaks=k)

optimizer = torch.optim.Adam(model.masses, lr=.01)
criterion = CrossEntropyLoss()

losses = []

print model

ti = time.time()
model.train()
epoch = 0
Xt = Variable(torch.Tensor(X_train))
yt = Variable(torch.Tensor(y_train).long())
# yt = torch.Tensor(y_train).view(len(y_train), 1)
# yt = Variable(torch.cat([yt == 0, yt == 1], 1).float())

for epoch in range(1000):
    print "Processing epoch %d" % (epoch + 1)
    y_pred = model.forward(Xt)
    loss = criterion(y_pred, yt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.normalize()
    losses.append(loss.data.item())
    if epoch > 2 and abs(losses[-2] - loss.data.item()) < 0.001:
        break

dt = time.time() - ti
print "Training time: %.2fs, epochs: %d" % (dt, epoch)
print "Least training loss reached: %.3f" % losses[-1]
model.eval()
print model

# TESTING
with torch.no_grad():
    Xt = torch.Tensor(X_test)
    Yt = torch.Tensor(y_test).long().numpy()
    _, yt_pred = torch.max(model(Xt), 1)
    yt_pred = yt_pred.numpy()
    accuracy = accuracy_score(Yt, yt_pred)
    print "Accuracy in test: %.1f%%" % (accuracy * 100)
    print "Confusion Matrix"
    print confusion_matrix(Yt, yt_pred)


# plt.scatter(X_test[:,0], X_test[:,1], c=yt_pred)
# plt.show()

plt.plot(range(len(losses)), losses)
plt.xlabel("Iterations")
plt.ylabel("CE")
# plt.axis([0, len(losses) - 1, losses[-1] - 0.05, losses[0] + 0.02])
plt.title("Error")
# plt.show()

print "N,k,m,Ac,time,epochs"
print "%d,%d,%d,%.3f,%.2f,%d" % (N, k, m, accuracy, dt, epoch)

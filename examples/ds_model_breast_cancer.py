import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from ds.DSModel import DSModel

data = pd.read_csv("data/breast-cancer-wisconsin.csv")

data = data.drop("id", axis=1)
data["class"] = data["class"].map({2: 0, 4: 1})

data = data.apply(pd.to_numeric, args=("coerce",))
data = data.sample(frac=1).reset_index(drop=True)

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].as_matrix()
y_train = data.iloc[:cut, -1].as_matrix()
X_test = data.iloc[cut:, :-1].as_matrix()
y_test = data.iloc[cut:, -1].as_matrix()

model = DSModel()
model.generate_statistic_rules(X_train, breaks=4)

optimizer = torch.optim.Adam(model.masses, lr=0.005)
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
    y_pred = model.forward(Xt)
    loss = criterion(y_pred, yt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.normalize()
    losses.append(loss.data.item())
    if epoch > 3 and abs(losses[-3] - loss.data.item()) < 0.0005:
        break

print "Training time: %.2fs, epochs: %d" % (time.time() - ti, epoch)
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


plt.plot(range(len(losses)), losses)
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy")
plt.axis([0, len(losses) - 1, losses[-1] - 0.05, losses[0] + 0.02])
plt.title("Error")
plt.show()
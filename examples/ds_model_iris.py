import pandas as pd
import time
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from ds.DSModel import DSModel

data = pd.read_csv("data/iris.csv")
data = data[data.species != "setosa"].reset_index(drop=True)  # Remove virginica to make the problem binary
data = data.sample(frac=1).reset_index(drop=True)

data = data.replace("versicolor", 0).replace("virginica", 1)

data = data.apply(pd.to_numeric)
cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].as_matrix()
y_train = data.iloc[:cut, -1].as_matrix()
X_test = data.iloc[cut:, :-1].as_matrix()
y_test = data.iloc[cut:, -1].as_matrix()

model = DSModel()
model.generate_statistic_single_rules(X_train, breaks=4)
model.generate_mult_pair_rules(X_train)
# print model
# exit()

# model.add_rule(lambda x: x[0] >= 6.)
# model.add_rule(lambda x: x[0] <= 5.)
# model.add_rule(lambda x: 5. < x[0] < 6.)
#
# model.add_rule(lambda x: x[1] >= 3.6)
# model.add_rule(lambda x: x[1] <= 3.)
# model.add_rule(lambda x: 3. < x[1] < 3.6)
#
# model.add_rule(lambda x: x[2] >= 5.)
# model.add_rule(lambda x: x[2] <= 2.)
# model.add_rule(lambda x: 2. < x[2] < 5.)
#
# model.add_rule(lambda x: x[3] >= 1.5)
# model.add_rule(lambda x: x[3] <= 1.)
# model.add_rule(lambda x: 1. < x[3] < 1.5)

optimizer = torch.optim.Adam(model.masses, lr=0.01)
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
    if epoch > 3 and abs(losses[-3] - loss.data.item()) < 0.001:
        break

print "Training time: %.3fs, epochs: %d" % (time.time() - ti, epoch)
print "Least training loss reached: %.3f" % losses[-1]
model.eval()
print model

# TESTING
with torch.no_grad():
    Xt = torch.Tensor(X_test)
    Yt = torch.Tensor(y_test).long()
    _, yt_pred = torch.max(model(Xt), 1)
    print Yt
    print yt_pred
    accuracy = (yt_pred.int() == Yt.int()).sum().item() / float(len(Yt))
    print "Accuracy in test: %.1f%%" % (accuracy * 100)

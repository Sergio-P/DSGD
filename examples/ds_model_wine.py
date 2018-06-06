import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from ds.DSModel import DSModel

data = pd.read_csv("data/wine.csv")
# data = pd.concat([data[data["color"] == "red"].sample(n=1200),
#                   data[data["color"] == "white"].sample(n=1200)])\
#         .reset_index(drop=True)


data["color"] = data["color"].map({"red": 0, "white": 1})

data = data.apply(pd.to_numeric, args=("coerce",))
data = data.sample(frac=1).reset_index(drop=True)

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values

model = DSModel()
model.generate_statistic_single_rules(X_train, breaks=2)
model.generate_mult_pair_rules(X_train)

optimizer = torch.optim.Adam(model.masses, lr=.05)
criterion = CrossEntropyLoss()

losses = []

# print model

ti = time.time()
model.train()
epoch = 0
Xt = Variable(torch.Tensor(X_train))
yt = Variable(torch.Tensor(y_train).long())
# yt = torch.Tensor(y_train).view(len(y_train), 1)
# yt = Variable(torch.cat([yt == 0, yt == 1], 1).float())

for epoch in range(10):
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

print "Training time: %.2fs, epochs: %d" % (time.time() - ti, epoch)
print "Least training loss reached: %.3f" % losses[-1]
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


plt.plot(range(len(losses)), losses)
plt.xlabel("Iterations")
plt.ylabel("CE")
# plt.axis([0, len(losses) - 1, losses[-1] - 0.05, losses[0] + 0.02])
plt.title("Error")
plt.savefig("wine_error.png")

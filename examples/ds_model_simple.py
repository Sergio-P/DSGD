import time
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import MSELoss, CrossEntropyLoss
from ds.DSModel import DSModel


model = DSModel()
model.add_rule(lambda x: x[0] > 1.5)
model.add_rule(lambda x: x[0] < -1)
model.add_rule(lambda x: -1.5 <= x[0] <= 1.5)
model.add_rule(lambda x: -1 <= x[0] <= 1)
model.add_rule(lambda x: -.5 <= x[0] <= .5)
model.add_rule(lambda x: 0 <= x[0])

optimizer = torch.optim.Adam(model.masses, lr=0.01)
criterion = CrossEntropyLoss()

losses = []

SIZE = 50
X = torch.rand(SIZE, 1) * 5 - 2.5
Yn = (X > -1.2) & (X < 1.2)
Y = torch.cat([Yn == 0, Yn == 1], 1).long()

y = Variable(Yn.view(SIZE).long())
X = Variable(X)

print(model)

ti = time.time()
model.train()
epoch = 0
for epoch in range(1000):
    y_pred = model.forward(X)
    # print y_pred.size()
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.normalize()
    losses.append(loss.data.item())
    if epoch > 3 and abs(losses[-3] - loss.data.item()) < 0.001:
        break

print("Training time: %.3fs, epochs: %d" % (time.time() - ti, epoch))
model.eval()
print(model)

# Testing
with torch.no_grad():
    SIZE = 50
    Xt = torch.rand(SIZE, 1) * 5 - 2.5
    Yt = ((Xt > -1.2) & (Xt < 1.2)).view(SIZE)
    _, yt_pred = torch.max(model(Xt), 1)
    accuracy = (yt_pred.int() == Yt.int()).sum().item() / float(len(Yt))
    print("Accuracy in test: %.1f%%" % (accuracy * 100))

plt.plot(range(len(losses)), losses)
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy")
plt.axis([0, len(losses) - 1, 0.0, losses[0] + 0.05])
plt.title("Error")
plt.show()

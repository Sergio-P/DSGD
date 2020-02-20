import torch
from torch import nn
from torch.autograd import Variable

from dsgd.core import dempster_rule_t


class DSClassifierGDSimple(nn.Module):
    def __init__(self):
        super(DSClassifierGDSimple, self).__init__()
        self.m = Variable(torch.rand(3, 1) * 0.5, requires_grad=True)
        self.normalize()

    def forward(self, X):
        out = []
        for i in range(len(X)):
            out.append(self.m[:2] / torch.sum(self.m[:2]))
        return torch.cat(out).view(len(X), 2)

    def normalize(self):
        self.m.data.clamp_(0., 1.)
        self.m.data.div_(torch.sum(self.m.data))


class DSClassifierGDDouble(nn.Module):
    def __init__(self):
        super(DSClassifierGDDouble, self).__init__()
        self.m1 = Variable(torch.rand(3, 1) * 0.5, requires_grad=True)
        self.m2 = Variable(torch.rand(3, 1) * 0.5, requires_grad=True)
        self.normalize()

    def forward(self, X):
        out = []
        for i in range(len(X)):
            mf = dempster_rule_t(self.m1, self.m2)
            out.append(mf[:2] / torch.sum(mf[:2]))
        return torch.cat(out).view(len(X), 2)

    def normalize(self):
        self.m1.data.div_(torch.sum(self.m1.data))
        self.m2.data.div_(torch.sum(self.m2.data))


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    P = 0.1
    model = DSClassifierGDDouble()
    optimizer = torch.optim.Adam([model.m1, model.m2], lr=0.005)
    criterion = nn.MSELoss()  # Mean Squared Loss

    losses = []
    w11 = []
    w12 = []
    w13 = []
    w21 = []
    w22 = []
    w23 = []

    # print "Initial values for w"
    # print model.m

    w11.append(float(model.m1.data[0]))
    w12.append(float(model.m1.data[1]))
    w13.append(float(model.m1.data[2]))
    w21.append(float(model.m2.data[0]))
    w22.append(float(model.m2.data[1]))
    w23.append(float(model.m2.data[2]))

    def create_vect(P, n=30):
        a = torch.rand(n, 1)
        return torch.cat([a > P, a <= P], 1).float()


    y = Variable(create_vect(P))

    ti = time.time()
    model.train()
    for epoch in range(1000):
        y_pred = model.forward(y)
        # print y
        # print y_pred
        # break
        # print (y_pred - y).pow(2).sum()
        # break
        # loss = (y_pred - y).pow(2).sum()
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize()
        losses.append(loss.data[0])
        w11.append(float(model.m1.data[0]))
        w12.append(float(model.m1.data[1]))
        w13.append(float(model.m1.data[2]))
        w21.append(float(model.m2.data[0]))
        w22.append(float(model.m2.data[1]))
        w23.append(float(model.m2.data[2]))
        if epoch > 3 and abs(losses[-3] - loss.data[0]) < 0.001:
            break

    print("Optimization time: %ds" % (time.time() - ti))
    model.eval()
    # print "Optimal values for w"
    # print model.m

    plt.subplot(131)
    plt.plot(range(len(w11)), w11, label="A")
    plt.plot(range(len(w12)), w12, label="B")
    plt.plot(range(len(w13)), w13, label="A,B")
    plt.legend()
    plt.axis([0, len(w11) - 1, -0.02, 1.0])
    plt.xlabel("Iterations")
    plt.ylabel("Mass values")
    plt.title("Mass optimization for m1")

    plt.subplot(132)
    plt.plot(range(len(w21)), w21, label="A")
    plt.plot(range(len(w22)), w22, label="B")
    plt.plot(range(len(w23)), w23, label="A,B")
    plt.legend()
    plt.axis([0, len(w21) - 1, -0.02, 1.0])
    plt.xlabel("Iterations")
    plt.ylabel("Mass values")
    plt.title("Mass optimization for m2")

    plt.subplot(133)
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.axis([0, len(losses) - 1, 0.0, losses[0] + 0.05])
    plt.title("Error")
    plt.show()
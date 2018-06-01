# coding=utf-8
import time
import torch
from torch.autograd import Variable

from ds.DSModel import DSModel


class DSClassifier():

    def __init__(self, lr=0.01, max_iter=200, min_dloss=0.001, optim="adam", lossfn="CE", debug_mode=False):
        self.lr = lr
        self.optim = optim
        self.lossfn = lossfn
        self.max_iter = max_iter
        self.min_dJ = min_dloss
        self.debug_mode = debug_mode
        self.model = DSModel()

    def fit(self, X, y, add_single_rules=False, single_rules_breaks=2, add_mult_rules=False, **kwargs):
        if add_single_rules:
            self.model.generate_statistic_single_rules(X, breaks=single_rules_breaks)
        if add_mult_rules:
            self.model.generate_mult_pair_rules(X)

        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.model.masses, lr=self.lr)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(self.model.masses, lr=self.lr)
        else:
            raise RuntimeError("Unknown optimizer %s" % self.optim)

        if self.lossfn == "CE":
            criterion = torch.nn.CrossEntropyLoss()
        elif self.lossfn == "MSE":
            criterion = torch.nn.MSELoss()
        else:
            raise RuntimeError("Unknown loss function %s" % self.lossfn)

        if self.debug_mode:
            return self.optimize_debug(X, y, optimizer, criterion, **kwargs)
        else:
            return self.optimize(X, y, optimizer, criterion,)

    def optimize(self, X, y, optimizer, criterion):
        losses = []
        self.model.train()

        Xt = Variable(torch.Tensor(X))
        if self.lossfn == "CE":
            yt = Variable(torch.Tensor(y).long())
        else:
            yt = torch.Tensor(y).view(len(y), 1)
            yt = Variable(torch.cat([yt == 0, yt == 1], 1).float())

        for epoch in range(self.max_iter):
            y_pred = self.model.forward(Xt)
            loss = criterion(y_pred, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.model.normalize()

            losses.append(loss.data.item())
            if epoch > 2 and abs(losses[-2] - loss.data.item()) < self.min_dJ:
                break

    def optimize_debug(self, X, y, optimizer, criterion, print_init_model=False, print_final_model=False, print_time=True,
                       print_partial_time=False, print_every_epochs=None, print_least_loss=True, return_patial_dt=False):
        losses = []

        if print_init_model:
            print self.model

        dt_forward = 0
        dt_loss = 0
        dt_optim = 0
        dt_norm = 0
        ti = time.time()

        self.model.train()
        epoch = 0
        Xt = Variable(torch.Tensor(X))

        if self.lossfn == "CE":
            yt = Variable(torch.Tensor(y).long())
        else:
            yt = torch.Tensor(y).view(len(y), 1)
            yt = Variable(torch.cat([yt == 0, yt == 1], 1).float())

        for epoch in range(self.max_iter):
            if print_every_epochs is not None and epoch % print_every_epochs == 0:
                print "Processing epoch %d" % (epoch + 1)

            tq = time.time()
            y_pred = self.model.forward(Xt)
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
            self.model.normalize()
            dt_norm += time.time() - tq

            losses.append(loss.data.item())
            if epoch > 2 and abs(losses[-2] - loss.data.item()) < self.min_dJ:
                break

        dt = time.time() - ti
        if print_time:
            print "\nTraining time: %.2fs, epochs: %d" % (dt, epoch)

        if print_partial_time:
            print "├- Forward eval time:  %.3fs" % dt_forward
            print "├- Loss backward time: %.3fs" % dt_loss
            print "├- Optimization time:  %.3fs" % dt_optim
            print "└- Normalization time: %.3fs" % dt_norm

        if print_least_loss:
            print "\nLeast training loss reached: %.3f" % losses[-1]

        if print_final_model:
            print self.model

        if return_patial_dt:
            return losses, epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm
        else:
            return losses, epoch, dt

    def predict(self, X, one_hot=False):
        self.model.eval()

        with torch.no_grad():
            Xt = torch.Tensor(X)
            if one_hot:
                return self.model(Xt).numpy()
            else:
                _, yt_pred = torch.max(self.model(Xt), 1)
                yt_pred = yt_pred.numpy()
                return yt_pred

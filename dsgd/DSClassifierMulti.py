# coding=utf-8
import time
import numpy as np
import pandas as pd
import torch
from sklearn.base import ClassifierMixin
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler

from dsgd.DSModelMulti import DSModelMulti


class DSClassifierMulti(ClassifierMixin):
    """
    Implementation of Classifier based on DSModel
    """

    def __init__(self, num_classes, lr=0.005, max_iter=200, min_iter=2, min_dloss=0.001, optim="adam", lossfn="MSE", debug_mode=False,
                 use_softmax=False, skip_dr_norm=True, batch_size=4000, num_workers=1, precompute_rules=False):
        """
        Creates the classifier and the DSModel (accesible in attribute model)
        :param lr: Learning rate
        :param max_iter: Maximun number of epochs in training
        :param min_dloss: Minium variation of loss to consider converged
        :param optim: [ adam | sgd ] Optimization Method
        :param lossfn: [ CE | MSE ] Loss function
        :param debug_mode: Enables debug in training (prtinting and output metrics)
        """
        self.k = num_classes
        self.lr = lr
        self.optim = optim
        self.lossfn = lossfn
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_dJ = min_dloss
        self.balance_class_data = False
        self.debug_mode = debug_mode
        self.model = DSModelMulti(num_classes, use_softmax=use_softmax, skip_dr_norm=skip_dr_norm, precompute_rules=precompute_rules)

    def fit(self, X, y, add_single_rules=False, single_rules_breaks=2, add_mult_rules=False, column_names=None, **kwargs):
        """
        Fits the model masses using gradient descent optimization
        :param X: Features for training
        :param y: Labels of features
        :param add_single_rules: Generates single rules
        :param single_rules_breaks: Single rule breaks number
        :param add_mult_rules: Generates multiplication pair rules
        :param kwargs: In case of debugging, parameters of optimize_debug
        """
        # if self.balance_class_data:
        #     nmin = np.max(np.bincount(y))
        #     Xm = pd.DataFrame(np.concatenate((X,y.reshape(-1,1)), axis=1))
        #     Xm = pd.concat([Xm[Xm.iloc[:,-1] == 0].sample(n=nmin, replace=True),
        #                     Xm[Xm.iloc[:,-1] == 1].sample(n=nmin, replace=True)], axis=0)\
        #             .sample(frac=1).reset_index(drop=True).values
        #     X = Xm[:,:-1]
        #     y = Xm[:,-1].astype(int)

        if add_single_rules:
            self.model.generate_statistic_single_rules(X, breaks=single_rules_breaks, column_names=column_names)
        if add_mult_rules:
            self.model.generate_mult_pair_rules(X, column_names=column_names)

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

        # Add index to X
        X = np.insert(X, 0, values=np.arange(0, len(X)), axis=1)
        # print(X)

        if self.debug_mode:
            return self._optimize_debug(X, y, optimizer, criterion, **kwargs)
        else:
            return self._optimize(X, y, optimizer, criterion, )

    def _optimize(self, X, y, optimizer, criterion):
        losses = []
        self.model.train()
        self.model.clear_rmap()

        Xt = Variable(torch.Tensor(X))
        if self.lossfn == "CE":
            yt = Variable(torch.LongTensor(y))
        else:
            yt = torch.nn.functional.one_hot(torch.LongTensor(y), self.k).float()

        dataset = torch.utils.data.TensorDataset(Xt, yt)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers, pin_memory=False)
        epoch = 0
        for epoch in range(self.max_iter):
            acc_loss = 0
            for Xi, yi in train_loader:
                y_pred = self.model.forward(Xi)
                loss = criterion(y_pred, yi)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.model.normalize()
                acc_loss += loss.data.item()

            losses.append(acc_loss)
            if epoch > self.min_iter and abs(losses[-2] - acc_loss) < self.min_dJ:
                break

        return losses, epoch

    def _optimize_debug(self, X, y, optimizer, criterion, print_init_model=False, print_final_model=False, print_time=True,
                        print_partial_time=False, print_every_epochs=None, print_least_loss=True, return_partial_dt=False,
                        disable_all_print=False, print_epoch_progress=False):
        losses = []
        print("Optimization started")

        if disable_all_print:
            print_every_epochs = None
            print_final_model = False
            print_partial_time = False
            print_time = False
            print_least_loss = False
            print_init_model = False

        if print_init_model:
            print(self.model)

        dt_forward = 0
        dt_loss = 0
        dt_optim = 0
        dt_norm = 0
        ti = time.time()

        self.model.train()
        self.model.clear_rmap()
        Xt = Variable(torch.Tensor(X))
        if self.lossfn == "CE":
            yt = Variable(torch.LongTensor(y))
        else:
            yt = torch.nn.functional.one_hot(torch.LongTensor(y), self.k).float()

        dataset = torch.utils.data.TensorDataset(Xt, yt)
        N = len(dataset)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                                   num_workers=self.num_workers, pin_memory=False)

        epoch = 0
        for epoch in range(self.max_iter):
            if print_every_epochs is not None and epoch % print_every_epochs == 0:
                print("\rProcessing epoch\t%d\t%.4f\t" % (epoch + 1, losses[-1] if len(losses) > 0 else 1), end="")
            acc_loss = 0
            if print_epoch_progress:
                acc_n = 0
                print("")
            for Xi, yi in train_loader:
                ni = len(yi)
                if print_epoch_progress:
                    acc_n += ni
                    print(("\r %d%% [" % (100*acc_n/N)) + "#"*int(25*acc_n/N) + " "*int(25 - 25*acc_n/N) + "]", end="", flush=True)
                tq = time.time()
                y_pred = self.model.forward(Xi)
                dt_forward += time.time() - tq

                tq = time.time()
                loss = criterion(y_pred, yi)
                if np.isnan(loss.data.item()):
                    print(self.model)
                    print(y_pred)
                    print(yi)
                    raise RuntimeError("Loss is NaN")
                optimizer.zero_grad()
                loss.backward()
                dt_loss += time.time() - tq

                tq = time.time()
                optimizer.step()
                dt_optim += time.time() - tq

                tq = time.time()
                self.model.normalize()
                dt_norm += time.time() - tq

                acc_loss += loss.data.item() * ni / N

            losses.append(acc_loss)
            if epoch > self.min_iter and abs(losses[-2] - acc_loss) < self.min_dJ:
                break

        dt = time.time() - ti
        if print_time:
            print("\nTraining time: %.2fs, epochs: %d" % (dt, epoch + 1))

        if print_partial_time:
            print("├- Forward eval time:  %.3fs" % dt_forward)
            print("├- Loss backward time: %.3fs" % dt_loss)
            print("├- Optimization time:  %.3fs" % dt_optim)
            print("└- Normalization time: %.3fs" % dt_norm)

        if print_least_loss:
            print("\nLeast training loss reached: %.3f" % losses[-1])

        if print_final_model:
            print(self.model)

        if return_partial_dt:
            return losses, epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm
        else:
            return losses, epoch, dt

    def predict(self, X, one_hot=False):
        """
        Predict the classes for the feature vectors
        :param X: Feature vectors
        :param one_hot: If true, it is returned the score of belogning to each class
        :return: Classes for each feature vector
        """
        self.model.eval()
        self.model.clear_rmap()
        X = np.insert(X, 0, values=np.arange(0, len(X)), axis=1)

        with torch.no_grad():
            Xt = torch.Tensor(X)
            if one_hot:
                return self.model(Xt).numpy()
            else:
                _, yt_pred = torch.max(self.model(Xt), 1)
                yt_pred = yt_pred.numpy()
                return yt_pred

    def predict_proba(self, X):
        """
        Predict the score of belogning to all classes
        :param X: Feature vector
        :return: Class scores for each feature vector
        """
        return self.predict(X, one_hot=True)

    def predict_explain(self, x):
        """
        Predict the score of belogning to each class and give an explanation of that decision
        :param x: A single Feature vectors
        :return:
        """
        pred = self.predict_proba([x])[0]
        cls = np.argmax(pred)
        rls = self.model.get_rules_by_instance(x, order_by=cls)

        # String interpretation
        builder = "DS Model predicts class %d\n" % cls
        for i in range(len(pred)-1):
            builder += " Class %d: \t%.3f\n" % (i, pred[i])
        builder += " Uncertainty:\t%.3f\n\n" % pred[-1]
        for i in range(min(len(rls), 5)):
            builder += " "
        return pred, cls, rls, builder



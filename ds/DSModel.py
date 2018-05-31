import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from scipy.stats import norm

from ds.DSRule import DSRule
from ds.core import dempster_rule_t, create_random_maf


class DSModel(nn.Module):
    """
    Torch module implementation of DS Binary Classifier
    """

    def __init__(self):
        """
        Creates an empty DS Model
        """
        super(DSModel, self).__init__()
        self.masses = []
        self.preds = []
        self.n = 0

    def add_rule(self, pred, ma=None, mb=None, mab=None):
        """
        Adds a rule to the model. If no masses are provided, random masses will be used.
        :param pred: DSRule or lambda or callable, used as the predicate of the rule
        :param ma: [optional] mass for first element
        :param mb: [optional] mass for second element
        :param mab: [optional] mass for uncertainty
        :return:
        """
        self.preds.append(pred)
        self.n += 1
        if ma is None or mb is None or mab is None:
            _, ma, mb, mab = create_random_maf()
        self.masses.append(Variable(torch.Tensor([[ma], [mb], [mab]]), requires_grad=True))

    def generate_statistic_rules(self, X, breaks=2):
        """
        Populates the model with attribute-independant rules based on statistical breaks
        :param X: Set of inputs (can be the same as training or a sample)
        :param breaks: Number of breaks per attribute
        """
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        brks = norm.ppf(np.linspace(0,1,breaks+2))[1:-1]
        for i in range(len(mean)):
            # First rule
            v = mean[i] + std[i] * brks[0]
            self.add_rule(DSRule(lambda x: x[i] <= v, "X[%d] < %.3f" % (i, v)))
            # Mid rules
            for j in range(1, len(brks)):
                vl = v
                v = mean[i] + std[i] * brks[j]
                self.add_rule(DSRule(lambda x: vl <= x[i] < v, "%.3f < X[%d] < %.3f" % (vl, i, v)))
            # Last rule
            self.add_rule(DSRule(lambda x: x[i] > v, "X[%d] > %.3f" % (i, v)))

    def forward(self, X):
        """
        Defines the computation performed at every call. Applying Dempster Rule for combining.
        :param X: Set of inputs
        :return: Set of prediction for each input in one hot encoding format
        """
        out = []
        for i in range(len(X)):
            sel = self._select_rules(X[i])
            if len(sel) == 0:
                raise RuntimeError("No rule especified for input No %d" % i)
            else:
                mf = self.masses[sel[0]]
                for j in range(1, len(sel)):
                    mf = dempster_rule_t(mf, self.masses[sel[j]])
                res = (mf[:2] / torch.sum(mf[:2])).view(2)
                out.append(res)
        return torch.cat(out).view(len(X), 2)

    def normalize(self):
        """
        Normalize all masses in order to keep constraints of DS
        """
        for mass in self.masses:
            mass.data.clamp_(0., 1.)
            mass.data.div_(torch.sum(mass.data))

    def _select_rules(self, x):
        x = x.data.numpy()
        sel = []
        for i in range(self.n):
            if self.preds[i](x):
                sel.append(i)
        return sel

    def extra_repr(self):
        """
        Shows the rules and their mass values
        :return: An string cointaing the information about rules
        """
        builder = "DS Classifier using %d rules\n" % self.n
        for i in range(self.n):
            ps = str(self.preds[i])
            ms = self.masses[i]
            builder += "\nRule %d: %s\n\t A: %.3f\t B: %.3f\tA,B: %.3f\n" % (i+1, ps, ms[0], ms[1], ms[2])
        return builder[:-1]
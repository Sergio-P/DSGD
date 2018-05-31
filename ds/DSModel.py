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
        :return: A string cointaing the information about rules
        """
        builder = "DS Classifier using %d rules\n" % self.n
        for i in range(self.n):
            ps = str(self.preds[i])
            ms = self.masses[i]
            builder += "\nRule %d: %s\n\t A: %.3f\t B: %.3f\tA,B: %.3f\n" % (i+1, ps, ms[0], ms[1], ms[2])
        return builder[:-1]

    def find_most_important_rules(self, classes=None, threshold=0.2):
        """
        Shows the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return: A string containing the information about most important rules
        """
        builder = "Most important rules\n"
        if classes is None:
            classes = [0, 1]
        for cls in classes:
            builder += " For class %d\n" % cls
            found = False
            for i in range(len(self.masses)):
                ms = self.masses[i]
                s = ((.1 + ms[cls]) / (.1 + ms[1 - cls]) - 1) * (1 - ms[-1]) / 10.
                score = np.sign(s) * np.sqrt(np.abs(s))
                if score >= threshold:
                    found = True
                    ps = str(self.preds[i])
                    builder += "  Rule %d: %s\n\t A: %.3f\t B: %.3f\tA,B: %.3f\n" % (i + 1, ps, ms[0], ms[1], ms[2])
            if not found:
                builder += "   No rules found\n"
        return builder[:-1]

    def generate_statistic_single_rules(self, X, breaks=2):
        """
        Populates the model with attribute-independant rules based on statistical breaks.
        In total this method generates No. attributes * (breaks + 1) rules
        :param X: Set of inputs (can be the same as training or a sample)
        :param breaks: Number of breaks per attribute
        """
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        brks = norm.ppf(np.linspace(0,1,breaks+2))[1:-1]
        for i in range(len(mean)):
            # First rule
            v = mean[i] + std[i] * brks[0]
            self.add_rule(DSRule(lambda x, i=i: x[i] <= v, "X[%d] < %.3f" % (i, v)))
            # Mid rules
            for j in range(1, len(brks)):
                vl = v
                v = mean[i] + std[i] * brks[j]
                self.add_rule(DSRule(lambda x, i=i: vl <= x[i] < v, "%.3f < X[%d] < %.3f" % (vl, i, v)))
            # Last rule
            self.add_rule(DSRule(lambda x, i=i: x[i] > v, "X[%d] > %.3f" % (i, v)))

    def generate_mult_pair_rules(self, X):
        """
        Populates the model with with rules combining 2 attributes by their multipication, adding both positive
        and negative rule. In total this method generates (No. attributes)^2 rules
        :param X: Set of inputs (can be the same as training or a sample)
        """
        mean = np.nanmean(X, axis=0)
        for i in range(len(mean)):
            for j in range(i+1,len(mean)):
                mk = mean[i] * mean[j]
                self.add_rule(DSRule(lambda x, i=i, j=j: x[i] * x[j] > mk,  "X[%d]*X[%d] > %.3f" % (i,j,mk)))
                self.add_rule(DSRule(lambda x, i=i, j=j: x[i] * x[j] <= mk, "X[%d]*X[%d] < %.3f" % (i,j,mk)))

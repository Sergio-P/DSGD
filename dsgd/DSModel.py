import torch
# import dill
import pickle
from torch import nn
from torch.autograd import Variable
import numpy as np
from scipy.stats import norm
from torch.nn import Softmax

from dsgd.DSRule import DSRule
from dsgd.core import dempster_rule_t, create_random_maf
from dsgd.utils import is_categorical


class DSModel(nn.Module):
    """
    Torch module implementation of DS Binary Classifier
    """

    def __init__(self, use_softmax=True, skip_dr_norm=False):
        """
        Creates an empty DS Model
        """
        super(DSModel, self).__init__()
        self.masses = []
        self.preds = []
        self.n = 0
        self.use_softmax = use_softmax
        self.skip_dr_norm = skip_dr_norm
        if use_softmax:
            self.sm = Softmax(dim=0)

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
                    mf = dempster_rule_t(mf, self.masses[sel[j]], not self.skip_dr_norm)
                if self.use_softmax:
                    res = self.sm(mf[:2])
                else:
                    res = (mf[:2] / torch.sum(mf[:2])).view(2)
                out.append(res)
        return torch.cat(out).view(len(X), 2)

    def normalize(self):
        """
        Normalize all masses in order to keep constraints of DS
        """
        for mass in self.masses:
            mass.data.clamp_(0., 1.)
            if self.use_softmax:
                mass = self.sm(mass)
            else:
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

    def find_most_important_rules(self, classes=None, threshold=0.2, class_names=None):
        """
        Shows the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return: A string containing the information about most important rules
        """
        builder = "Most important rules\n"
        if classes is None:
            classes = [0, 1]

        if class_names is None:
            class_names = [str(i) for i in range(2)]

        for cls in classes:
            builder += "\n For class %s\n" % class_names[cls]
            found = []
            for i in range(len(self.masses)):
                ms = self.masses[i]
                score = ((.1 + ms[cls]) / (.1 + ms[1 - cls]) - 1) * (1 - ms[-1]) / 10.
                if score >= threshold * threshold:
                    ps = str(self.preds[i])
                    found.append((score, "  Rule %d: %s (%.3f)\n\t A: %.3f\t B: %.3f\tA,B: %.3f\n" % \
                                (i + 1, ps, np.sqrt(score.detach().numpy()), ms[0], ms[1], ms[2])))

            found.sort(reverse=True)
            if len(found) == 0:
                builder += "   No rules found\n"

            for rule in found:
                builder += rule[1]

        return builder[:-1]

    def generate_statistic_single_rules(self, X, breaks=2, column_names=None):
        """
        Populates the model with attribute-independant rules based on statistical breaks.
        In total this method generates No. attributes * (breaks + 1) rules
        :param X: Set of inputs (can be the same as training or a sample)
        :param breaks: Number of breaks per attribute
        :param column_names: Column attribute names
        """
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        brks = norm.ppf(np.linspace(0,1,breaks+2))[1:-1]

        if column_names is None:
            column_names = ["X[%d]" % i for i in range(len(mean))]

        for i in range(len(mean)):
            if is_categorical(X[:,i]):
                categories = np.unique(X[:,i][~np.isnan(X[:,i])])
                if len(categories) <= 1:
                    continue
                for cat in categories:
                    self.add_rule(DSRule(lambda x, i=i, k=cat: x[i] == k, "%s = %s" % (column_names[i], str(cat))))
            else:
                # First rule
                v = mean[i] + std[i] * brks[0]
                self.add_rule(DSRule(lambda x, i=i, v=v: x[i] <= v, "%s < %.3f" % (column_names[i], v)))
                # Mid rules
                for j in range(1, len(brks)):
                    vl = v
                    v = mean[i] + std[i] * brks[j]
                    self.add_rule(DSRule(lambda x, i=i, vl=vl, v=v: vl <= x[i] < v, "%.3f < %s < %.3f" % (vl, column_names[i], v)))
                # Last rule
                self.add_rule(DSRule(lambda x, i=i, v=v: x[i] > v, "%s > %.3f" % (column_names[i], v)))

    def generate_categorical_rules(self, X, column_names=None, exclude=None):
        """
        Populates the model with attribute-independant rules based on categories of attributes, continous columns are
        skipped.
        :param X: Set of inputs (can be the same as training or a sample)
        :param column_names: Column attribute names
        """
        m = X.shape[1]
        if column_names is None:
            column_names = ["X[%d]" % i for i in range(m)]

        if exclude is None:
            exclude = []

        for i in range(m):
            if is_categorical(X[:,i]) and column_names[i] not in exclude:
                categories = np.unique(X[:,i][~np.isnan(X[:,i])])
                for cat in categories:
                    self.add_rule(DSRule(lambda x, i=i, k=cat: x[i] == k, "%s = %s" % (column_names[i], str(cat))))

    def generate_mult_pair_rules(self, X, column_names=None, include_square=False):
        """
        Populates the model with with rules combining 2 attributes by their multipication, adding both positive
        and negative rule. In total this method generates (No. attributes)^2 rules
        :param X: Set of inputs (can be the same as training or a sample)
        :param column_names: Column attribute names
        :param include_square: Includes rules comparing the same attribute (ie x[i] * x[i])
        """
        mean = np.nanmean(X, axis=0)

        if column_names is None:
            column_names = ["X[%d]" % i for i in range(len(mean))]

        offset = 0 if include_square else 1

        for i in range(len(mean)):
            for j in range(i + offset, len(mean)):
                # mk = mean[i] * mean[j]
                mi = mean[i]
                mj = mean[j]
                self.add_rule(DSRule(lambda x, i=i, j=j, mi=mi, mj=mj: (x[i] - mi) * (x[j] - mj) > 0,
                                     "Positive %s - %.3f, %s - %.3f" % (column_names[i],mean[i],column_names[j],mean[j])))
                self.add_rule(DSRule(lambda x, i=i, j=j, mi=mi, mj=mj: (x[i] - mi) * (x[j] - mj) <= 0,
                                     "Negative %s - %.3f, %s - %.3f" % (column_names[i],mean[i],column_names[j],mean[j])))

    def generate_custom_range_single_rules(self, column_names, name, breaks):
        """
        Populates the model with attribute-independant rules based on custom defined breaks.
        In total this method generates len(breaks) + 1 rules
        :param column_names: Column attribute names
        :param name: The target column name to generate rules
        :param breaks: Array of float indicating the values of the breaks
        """
        i = column_names.tolist().index(name)
        if i == -1:
            raise NameError("Cannot find column with name %s" % name)
        v = breaks[0]
        # First rule
        self.add_rule(DSRule(lambda x, i=i, v=v: x[i] <= v, "%s < %.3f" % (name, v)))
        # Mid rules
        for j in range(1, len(breaks)):
            vl = v
            v = breaks[j]
            self.add_rule(DSRule(lambda x, i=i, vl=vl, v=v: vl <= x[i] < v, "%.3f < %s < %.3f" % (vl, name, v)))
        # Last rule
        self.add_rule(DSRule(lambda x, i=i, v=v: x[i] > v, "%s > %.3f" % (name, v)))

    def generate_custom_range_rules_by_gender(self, column_names, name, breaks_men, breaks_women, gender_name="gender"):
        """
        Populates the model with attribute-independant rules based on custom defined breaks separated by gender.
        :param column_names: Column attribute names
        :param name: The target column name to generate rules
        :param breaks_men: Array of float indicating the values of the breaks for men
        :param breaks_women: Array of float indicating the values of the breaks for women
        :param gender_name: Name of the column containing the gender
        """
        i = column_names.tolist().index(name)
        g = column_names.tolist().index(gender_name)

        if i == -1 or g == -1:
            raise NameError("Cannot find column with name %s" % name)

        for gv, gname, breaks in [(0, "Men", breaks_men), (1, "Women", breaks_women)]:
            v = breaks[0]
            # First rule
            self.add_rule(DSRule(lambda x, i=i, g=g, gv=gv, v=v: x[g] == gv and x[i] <= v, "%s: %s < %.3f" % (gname, name, v)))
            # Mid rules
            for j in range(1, len(breaks)):
                vl = v
                v = breaks[j]
                self.add_rule(DSRule(lambda x, i=i, g=g, gv=gv, v=v: x[g] == gv and vl <= x[i] < v, "%s: %.3f < %s < %.3f" %
                                     (gname, vl, name, v)))
            # Last rule
            self.add_rule(DSRule(lambda x, i=i, g=g, gv=gv, v=v: x[g] == gv and x[i] > v, "%s: %s > %.3f" % (gname, name, v)))

    def generate_outside_range_pair_rules(self, column_names, ranges):
        """
        Populates the model with outside-normal-range pair of attributes rules
        :param column_names: The columns names in the dataset
        :param ranges: Matrix size (k,3) indicating the lower, the upper bound and the name of the column
        """
        for index_i in range(len(ranges)):
            col_i = ranges[index_i][2]
            i = column_names.tolist().index(col_i)
            for index_j in range(index_i + 1, len(ranges)):
                col_j = ranges[index_j][2]
                j = column_names.tolist().index(col_j)
                # Extract ranges
                li = ranges[index_i][0]
                hi = ranges[index_i][1]
                lj = ranges[index_j][0]
                hj = ranges[index_j][1]
                # Add Rules
                if not np.isnan(li) and not np.isnan(lj):
                    self.add_rule(DSRule(lambda x, i=i, j=j, li=li, lj=lj: x[i] < li and x[j] < lj, "Low %s and Low %s" % (col_i, col_j)))
                if not np.isnan(hi) and not np.isnan(lj):
                    self.add_rule(DSRule(lambda x, i=i, j=j, hi=hi, lj=lj: x[i] > hi and x[j] < lj, "High %s and Low %s" % (col_i, col_j)))
                if not np.isnan(hi) and not np.isnan(hj):
                    self.add_rule(DSRule(lambda x, i=i, j=j, hi=hi, hj=hj: x[i] > hi and x[j] > hj, "High %s and High %s" % (col_i, col_j)))
                if not np.isnan(li) and not np.isnan(hj):
                    self.add_rule(DSRule(lambda x, i=i, j=j, li=li, hj=hj: x[i] < li and x[j] > hj, "Low %s and High %s" % (col_i, col_j)))

    def load_rules_bin(self, filename):
        """
        Loads rules from a file, it deletes previous rules
        :param filename: The name of the input file
        """
        with open(filename) as f:
            sv = pickle.load(f)
            self.preds = sv["preds"]
            self.masses = sv["masses"]

        print(self.preds)

    def save_rules_bin(self, filename):
        """
        Saves the current rules into a file
        :param filename: The name of the file
        """
        with open(filename, "w") as f:
            sv = {"preds": self.preds, "masses": self.masses}
            pickle.dump(sv, f, pickle.HIGHEST_PROTOCOL)

    def get_rules_size(self):
        return self.n
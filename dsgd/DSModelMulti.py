import torch
# import dill
import pickle
from torch import nn
from torch.autograd import Variable
import numpy as np
from scipy.stats import norm
from torch.nn import Softmax

from dsgd.DSRule import DSRule
from dsgd.core import dempster_rule_kt, create_random_maf_k
from dsgd.utils import is_categorical


class DSModelMulti(nn.Module):
    """
    Torch module implementation of DS General Classification
    """

    def __init__(self, k, use_softmax=True, skip_dr_norm=False, precompute_rules=False):
        """
        Creates an empty DS Model
        """
        super(DSModelMulti, self).__init__()
        self.masses = []
        self.preds = []
        self.n = 0
        self.k = k
        self.use_softmax = use_softmax
        self.skip_dr_norm = skip_dr_norm
        self.precompute_rules = precompute_rules
        self.rmap = {}
        if use_softmax:
            self.sm = Softmax(dim=0)

    def add_rule(self, pred, m_sing=None, m_uncert=None):
        """
        Adds a rule to the model. If no masses are provided, random masses will be used.
        :param pred: DSRule or lambda or callable, used as the predicate of the rule
        :param m_sing: [optional] masses for singletons
        :param m_uncert: [optional] mass for uncertainty
        :return:
        """
        self.preds.append(pred)
        self.n += 1
        if m_sing is None or m_uncert is None or len(m_sing) != self.k:
            masses = create_random_maf_k(self.k)
        else:
            masses = m_sing + [m_uncert]
        self.masses.append(Variable(torch.Tensor(masses).view(self.k + 1, 1), requires_grad=True))

    def forward(self, X):
        """
        Defines the computation performed at every call. Applying Dempster Rule for combining.
        :param X: Set of inputs
        :return: Set of prediction for each input in one hot encoding format
        """
        out = torch.zeros(len(X), self.k)
        for i in range(len(X)):
            sel = self._select_rules(X[i, 1:], int(X[i, 0].item()))
            if len(sel) == 0:
                raise RuntimeError("No rule especified for input No %d" % i)
            else:
                mf = self.masses[sel[0]]
                for j in range(1, len(sel)):
                    mf = dempster_rule_kt(mf, self.masses[sel[j]], not self.skip_dr_norm)
                if self.use_softmax:
                    res = self.sm(mf[:-1])
                elif torch.sum(mf[:-1]) > 0:
                    res = (mf[:-1] / torch.sum(mf[:-1])).view(self.k)
                else:
                    print("Warning: Total conflict mass found")
                    print(X[i])
                    print(mf)
                    res = mf[:-1].view(self.k)
                out[i] = res
        return out

    def clear_rmap(self):
        self.rmap = {}

    def normalize(self):
        """
        Normalize all masses in order to keep constraints of DS
        """
        for mass in self.masses:
            mass.data.clamp_(0., 1.)
            if self.use_softmax:
                mass = self.sm(mass)
            else:
                if torch.sum(mass.data) == 0:
                    raise RuntimeError("Zero mass sum")
                mass.data.div_(torch.sum(mass.data))

    def _select_rules(self, x, index=None):
        if self.precompute_rules and index in self.rmap:
            return self.rmap[index]
        x = x.data.numpy()
        sel = []
        for i in range(self.n):
            if self.preds[i](x):
                sel.append(i)
        if self.precompute_rules and index is not None:
            self.rmap[index] = sel
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
            builder += "\nRule %d: %s\n\t" % (i+1, ps)
            for j in range(len(ms) - 1):
                builder += "C%d: %.3f\t" % (j+1, ms[j])
            builder += "Unc: %.3f\n" % ms[self.k]
        return builder[:-1]

    def find_most_important_rules(self, classes=None, threshold=0.2):
        """
        Shows the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return: A list containing the information about most important rules
        """
        if classes is None:
            classes = [i for i in range(self.k)]

        with torch.no_grad():
            rules = {}
            for j in range(len(classes)):
                cls = classes[j]
                found = []
                for i in range(len(self.masses)):
                    ms = self.masses[i].view(-1).detach().numpy()
                    score = (ms[j]) * (1 - ms[-1])
                    if score >= threshold * threshold:
                        ps = str(self.preds[i])
                        found.append((score, i, ps, np.sqrt(score), ms.tolist()))

                found.sort(reverse=True)
                rules[cls] = found

        return rules

    def print_most_important_rules(self, classes=None, threshold=0.2):
        rules = self.find_most_important_rules(classes, threshold)

        if classes is None:
            classes = [i for i in range(self.k)]

        builder = ""
        for i in range(len(classes)):
            rs = rules[classes[i]]
            builder += "\n\nMost important rules for class %s" % classes[i]
            for r in rs:
                builder += "\n\n\t[%.3f] R%d: %s\n\t\t" % (r[3], r[1], r[2])
                masses = r[4]
                for j in range(len(masses)):
                    builder += "\t%s: %.3f" % (classes[j][:3] if j < len(classes) else "Unc", masses[j])
        print(builder)

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
            self.add_rule(DSRule(lambda x, i=i, v=v, g=g, gv=gv: x[g] == gv and x[i] <= v, "%s: %s < %.3f" % (gname, name, v)))
            # Mid rules
            for j in range(1, len(breaks)):
                vl = v
                v = breaks[j]
                self.add_rule(DSRule(lambda x, i=i, g=g, gv=gv, vl=vl, v=v: x[g] == gv and vl <= x[i] < v, "%s: %.3f < %s < %.3f" %
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

    def get_rules_by_instance(self, x, order_by=0):
        sel = self._select_rules(x)
        rules = np.zeros((len(sel), self.k + 1))
        for i in range(len(sel)):
            rules[i, :] = self.masses[sel[i]].data.numpy()
        rules = rules[np.lexsort((rules[:, order_by],))]
        return rules

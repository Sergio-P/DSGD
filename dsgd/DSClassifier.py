import json
import pickle

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from core import create_full_uncertainty, cls_max, dempster_rule, cls_score, cls_pla_score, cls_pla_max
from dsgd.utils import statistic_breaks, is_categorical


class DSClassifier(ClassifierMixin):
    def __init__(self, columns=[]):
        self.rules = []
        self._columns = columns
        self._dec_fun = cls_score
        self._cls_fun = cls_max

    def fit(self, X, y, brks=5, stol=1):
        self.rules = []
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.1)
        X = pd.DataFrame(X, columns=self._columns)
        for col in self._columns:
            df = pd.DataFrame({"val": X[col], "cls": y})
            if is_categorical(df.val):
                for val in df.val.unique():
                    df_filter = df[df.val == val]
                    cv = pd.value_counts(df_filter["cls"])
                    n, ns, s, e = DSClassifier._construct_maf(cv, df_filter, df)
                    rule = Rule((n, ns, s, e), [Condition(col, val, val)])
                    self.rules.append(rule)
            else:
                breaks = statistic_breaks(df.val, k=brks, sigma_tol=stol, append_infinity=True)
                for i in xrange(len(breaks) - 1):
                    df_filter = df[(df.val >= breaks[i]) & (df.val <= breaks[i + 1])]
                    cv = pd.value_counts(df_filter["cls"])
                    n, ns, s, e = DSClassifier._construct_maf(cv, df_filter, df)
                    rule = Rule((n, ns, s, e), [Condition(col, breaks[i], breaks[i + 1])])
                    self.rules.append(rule)
        # VALIDATION
        y_pred = self._predict(X_val, cls_score, 0)
        sc_bel = roc_auc_score(y_val, y_pred)
        y_pred = self._predict(X_val, cls_pla_score, 0)
        sc_pla = roc_auc_score(y_val, y_pred)
        if sc_bel > sc_pla:
            self._dec_fun = cls_score
            self._cls_fun = cls_max
        else:
            self._dec_fun = cls_pla_score
            self._cls_fun = cls_pla_max

    def add_custom_rules(self, new_rules):
        self.rules.extend(new_rules)

    @staticmethod
    def _construct_maf(cv, df_filter, df):
        if len(df_filter) == 0:
            return 0, 0, 0, 1

        # Certainty computing
        d0 = float(cv[False] if False in cv else 0) / len(df_filter)
        d1 = float(cv[True] if True in cv else 0) / len(df_filter)

        # Uncertainty computing
        reg_factor = 1 - float(len(df_filter)) / len(df)
        dif_factor = 0 if max(d0, d1) == 0 else 0.5 * (1 - (max(d0, d1) - min(d0, d1)) / max(d0, d1))
        # reg_factor = 0

        # Mass assigment
        k = 1 + reg_factor + dif_factor
        mf_null = 0
        mf_no_stroke = d0 / k
        mf_stroke = d1 / k
        mf_either = (reg_factor + dif_factor) / k
        return mf_null, mf_no_stroke, mf_stroke, mf_either

    def load_rules_bin(self, filename):
        with open(filename) as f:
            self.rules = pickle.load(f)

    def save_rules_bin(self, filename):
        with open(filename, "w") as f:
            pickle.dump(self.rules, f, pickle.HIGHEST_PROTOCOL)

    def show_rules(self, print_as_csv=False):
        for rule in self.rules:
            if not print_as_csv:
                rule.pretty_print()
            else:
                rule.print_row()

    def predict(self, X, threshold=0):
        return self._predict(X, self._cls_fun, threshold)

    def predict_proba(self, X, threshold=0):
        return self._predict(X, self._dec_fun, threshold)

    def _predict(self, X, f, threshold):
        y = []
        X = pd.DataFrame(X, columns=self._columns)
        for (idx, x) in X.iterrows():
            m = create_full_uncertainty()
            for rule in self.rules:
                if DSClassifier._satisfy_rule(rule, x):
                    m = dempster_rule(m, rule.maf)
            y.append(f(m, threshold))
        return y

    @staticmethod
    def _satisfy_rule(rule, reg):
        satisfy = True
        for cond in rule.conditions:
            satisfy = satisfy and cond.range_min <= reg[cond.column] <= cond.range_max
        return satisfy


class Rule:
    def __init__(self, maf, conds):
        self.conditions = conds
        self.maf = maf

    def to_dict(self):
        return self.__dict__

    def pretty_print(self):
        print("RULE")
        print("   * MAF: " + str(self.maf))
        print("   * Conditions:")
        for cond in self.conditions:
            print("      * " + str(cond))

    def print_row(self):
        print(" & ".join(map(str, self.conditions)) + "," + ",".join(map(str, self.maf)))


class Condition:
    def __init__(self, col, range_min, range_max):
        self.column = col
        self.range_min = range_min
        self.range_max = range_max

    def __repr__(self):
        return json.dumps(self.__dict__)

    def __str__(self):
        return "Column: %s, From: %.3f - To: %.3f" % (self.column, self.range_min, self.range_max)


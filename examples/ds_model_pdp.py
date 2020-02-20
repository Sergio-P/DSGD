import json
import time

import torch

import numpy as np
from pdpbox import pdp
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from dsgd.DSClassifierMultiQ import DSClassifierMultiQ

data = pd.read_csv("data/stroke_data_18.csv")

# data = data.drop("0", axis=1)
data = data.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=10)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=0.25).reset_index(drop=True)


X_test = data.iloc[:, :-1].values
y_test = data.iloc[:, -1].values


DSC = DSClassifierMultiQ(2, min_iter=50, max_iter=100, debug_mode=True, num_workers=4, lossfn="MSE", optim="adam",
                         precompute_rules=True, batch_size=200, lr=0.005)
DSC.model.load_rules_bin("stroke_single.dsb")

# y_score = DSC.predict_proba(X_test)
# _, y_pred = torch.max(torch.Tensor(y_score), 1)
# y_pred = y_pred.numpy()
#
# print("Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("AUC score: %.3f" % (roc_auc_score(y_test, y_score[:, 1])))

# print(data.columns[:-1].tolist())
# exit()

FEATURE = "PLT"
st = time.time()
pdp_attr = pdp.pdp_isolate(DSC, data.iloc[:, :-1], data.columns[:-1].tolist(), FEATURE, num_grid_points=8)
pdp.pdp_plot(pdp_attr, FEATURE, frac_to_plot=0.5, plot_lines=True, figsize=(8, 6), x_quantile=True,
             plot_params={"pdp_hl_color": "#fe8c00", "zero_color": "#000000"}, center=True)
plt.tight_layout()
plt.show()
print("Took %.2fs" % (time.time() - st))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data = pd.read_csv("stroke-last-pred.csv", header=None).values

n = len(data)/2
y_true = data[:n]
y_score = data[n:]

while True:
    print("-"*60)
    t = float(raw_input("Threshold (0-1): "))
    if t == -1: break
    y_pred = (y_score > t).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    print("Confusion Matrix:")
    print(cm)

    print(classification_report(y_true, y_pred, labels=[1,0], target_names=["Stroke", "No Stroke"]))

    print("Accuracy: \t%.1f" % (100. * accuracy_score(y_true, y_pred)))
    print("Sensitivity:\t%.1f" % (100. * cm[0,0] / (cm[0,0] + cm[0,1])))
    print("Specificity:\t%.1f" % (100. * cm[1,1] / (cm[1,1] + cm[1,0])))
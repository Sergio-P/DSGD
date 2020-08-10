import time
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from other.tabular import TabularDataset, TabularNN

data = pd.read_csv("data/stroke_data_18.csv")

data = data.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=10)
data = data.apply(pd.to_numeric, args=("coerce",))
data["cls"] = data["cls"].map({0.: 0, 1.: 1})
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# print(data.columns)

print("ALL")

cat_cols = ["gender", "protein_qualitative", "sugar_qualitative", "occult_blood_reaction",
            "HD_diabetes", 'HD_cerebrovascullar', 'HD_cardiovascullar', 'HD_arteries']
emb_dims = []
labels_map = {}

for col in data.columns:
    if col in cat_cols:
        labels_map[col] = LabelEncoder()
        data[col] = data[col].fillna(-1)
        n = int(len(data[col].unique()))
        data[col] = labels_map[col].fit_transform(data[col])
        emb_dims.append((n, n // 2))
    else:
        data[col] = data[col].fillna(data[col].median())

# print(list(zip(cat_cols, emb_dims)))
# exit()

continous_num = len(data.columns) - len(cat_cols) - 1
n_tr = int(len(data) * 0.7)

BATCH_SIZE = 200

dataset_train = TabularDataset(data[:n_tr], cat_cols, "cls")
dataloader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle=True, num_workers=1)

dataset_test = TabularDataset(data, cat_cols, "cls")
dataloader_test = DataLoader(dataset_test, BATCH_SIZE, shuffle=True, num_workers=1)

model = TabularNN(emb_dims=emb_dims, no_of_cont=continous_num, lin_layer_sizes=[100],
                  output_size=2, emb_dropout=0.04, lin_layer_dropouts=[0.01])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

NUM_EPOCHS = 200
ls = 1.

ti = time.time()

for epoch in range(NUM_EPOCHS):
    print("\rProcessing epoch %d \t Loss: %.4f" % (epoch + 1, ls), end="")
    ls = 0.
    for y, cont_x, cat_x in dataloader_train:
        # Forward Pass
        preds = model(cont_x, cat_x)
        loss = criterion(preds, y)
        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ls += loss.detach().item() * len(y) / len(data[n_tr:])

dt = time.time() - ti

model.eval()
y_test = []
y_pred = []
y_score = []

y_pred_train = []
y_train = []

for y, cont_x, cat_x in dataloader_train:
    preds = model(cont_x, cat_x)[:, 1]
    y_pred_train.extend((preds.detach().numpy() > 0.5).astype(np.int).tolist())
    _, yt = y.max(1)
    y_train.extend(yt.tolist())

print("\n\nAccuracy Train: %.1f%%" % (accuracy_score(y_train, y_pred_train) * 100.))

for y, cont_x, cat_x in dataloader_test:
    preds = model(cont_x, cat_x)[:, 1]
    y_score.extend(preds.tolist())
    y_pred.extend((preds.detach().numpy() > 0.5).astype(np.int).tolist())
    _, yt = y.max(1)
    y_test.extend(yt.tolist())

# print("\nEpochs: %d" % epoch)
# print("Min Loss: %.4f" % ls)
print("\n\nTraining Time: %.1f" % dt)
print("Num of parameters: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("AUC score: %.3f" % (roc_auc_score(y_test, y_score)))
# print(DSC.model.find_most_important_rules(threshold=0.2, class_names=["No Stroke", "Stroke"]))

# print("\nAccuracy: %.4f" % accuracy)

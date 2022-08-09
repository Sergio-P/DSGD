import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from dsgd.DSClassifierMultiQ import DSClassifierMultiQ


def main():

    data = pd.read_csv("data/digits.csv", header=None)
    data = data.rename(columns={64: "cls"})

    # data = data[data.cls <= 3].reset_index(drop=True)  # Only 0s and 1s
    data = data.sample(frac=1).reset_index(drop=True)
    cut = int(0.7*len(data))

    X_train = data.iloc[:cut, :-1].values
    X_train = (X_train >= 5).astype(int) + (X_train >= 10).astype(int)
    y_train = data.iloc[:cut, -1].values
    X_test = data.iloc[cut:, :-1].values
    X_test = (X_test >= 5).astype(int) + (X_test >= 10).astype(int)
    y_test = data.iloc[cut:, -1].values

    training_times = []
    # Optimizations

    # Optimizations 1: Force precomputa
    DSC = DSClassifierMultiQ(10, max_iter=100, lr=0.01, debug_mode=True,
                             lossfn="MSE", precompute_rules=True, batch_size=500,
                             force_precompute=True)
    losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=1, print_every_epochs=1,
                                print_partial_time=True, print_time=True)
    y_pred = DSC.predict(X_test)
    print("Total Rules: %d" % DSC.model.get_rules_size())
    print("\nTraining Time: %.2f" % dt)
    print("Epochs: %d" % (epoch + 1))
    print("Min Loss: %.3f" % losses[-1])
    print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
    print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))
    training_times.append(dt)

    # Optimizations 2: Use CUDA
    DSC = DSClassifierMultiQ(10, max_iter=100, lr=0.01, debug_mode=True,
                             lossfn="MSE", precompute_rules=True, batch_size=500,
                             force_precompute=False,
                             device="cuda")
    losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=1, print_every_epochs=1,
                                print_partial_time=True, print_time=True)
    y_pred = DSC.predict(X_test)
    print("Total Rules: %d" % DSC.model.get_rules_size())
    print("\nTraining Time: %.2f" % dt)
    print("Epochs: %d" % (epoch + 1))
    print("Min Loss: %.3f" % losses[-1])
    print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
    print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))
    training_times.append(dt)

    # Optimization 3: Use both
    DSC = DSClassifierMultiQ(10, max_iter=100, lr=0.01, debug_mode=True,
                             lossfn="MSE", precompute_rules=True, batch_size=500,
                             force_precompute=True,
                             device="cuda")
    losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=1, print_every_epochs=1,
                                print_partial_time=True, print_time=True)
    y_pred = DSC.predict(X_test)
    print("Total Rules: %d" % DSC.model.get_rules_size())
    print("\nTraining Time: %.2f" % dt)
    print("Epochs: %d" % (epoch + 1))
    print("Min Loss: %.3f" % losses[-1])
    print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
    print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))
    training_times.append(dt)

    # Non optimized
    DSC = DSClassifierMultiQ(10, max_iter=100, lr=0.01, debug_mode=True,
                             lossfn="MSE", precompute_rules=True, batch_size=500,
                             force_precompute=False,
                             device="cpu")
    losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=1, print_every_epochs=1,
                                print_partial_time=True, print_time=True)
    y_pred = DSC.predict(X_test)
    print("Total Rules: %d" % DSC.model.get_rules_size())
    print("\nTraining Time: %.2f" % dt)
    print("Epochs: %d" % (epoch + 1))
    print("Min Loss: %.3f" % losses[-1])
    print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
    print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))
    training_times.append(dt)

    # Plot
    plt.figure(figsize=(6, 4))

    plt.bar(["force_precompute", "CUDA", "both", "None"], training_times)
    plt.title('Training times')
    plt.show()


if __name__ == "__main__":
    main()

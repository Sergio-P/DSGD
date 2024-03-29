import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_score

from dsgd.DSClassifierMultiQ import DSClassifierMultiQ

def main():
    data = pd.read_csv("data/iris.csv")
    data = data.sample(frac=1).reset_index(drop=True)

    data = data.replace("setosa", 0).replace("virginica", 1).replace("versicolor", 2)

    data = data.apply(pd.to_numeric)
    cut = int(0.3 * len(data))

    X_train = data.iloc[:cut, :-1].values
    y_train = data.iloc[:cut, -1].values
    X_test = data.iloc[cut:, :-1].values
    y_test = data.iloc[cut:, -1].values

    DSC = DSClassifierMultiQ(3, min_iter=50, max_iter=400, debug_mode=True, lossfn="MSE", num_workers=0, min_dloss=1e-7)
    # DSC.model.load_rules_bin("rules_iris.dsb")

    losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=3, add_mult_rules=False,
                                column_names=data.columns[:-1], print_every_epochs=31, print_final_model=False)
    y_pred = DSC.predict(X_test)
    DSC.print_most_important_rules(classes=["setosa", "virginica", "versicolor"])
    # print("\nTraining Time: %.2f" % dt)
    # print("Epochs: %d" % (epoch + 1))
    # print("Min Loss: %.3f" % losses[-1])
    print("\nAccuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.))
    print("F1 Macro: %.3f" % (f1_score(y_test, y_pred, average="macro")))
    print("F1 Micro: %.3f" % (f1_score(y_test, y_pred, average="micro")))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Explaining instance: ")
    print(X_test[0])
    pred, cls, rls, builder = DSC.predict_explain(X_test[0])
    print(builder)
    print(rls)

    #
    # DSC.model.save_rules_bin("rules_iris.dsb")

if __name__ == "__main__":
    main()

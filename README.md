# DSGD

Tabular interpretable classifier based on Dempster-Shafer Theory and 
Gradient Descent

## Description

This repository contains 3 implementations of the classifier:

- `DSClassifier` for binary classification problems
- `DSClassifierMulti` for multi-class classification problems
- `DSClassifierMultiQ` for multi-class classification problems 
that also includes the commonality transformation improvement 
which makes computations faster

We always recommend using `DSClassifierMultiQ` since it is the 
most stable and fastest implementation. Multi-class implementations
can handle binary problems as well.

## Installation

TBD

## Usage

### Import the module

    from ds.DSClassifierMultiQ import DSClassifierMultiQ
    
### Read the data 

The data can be read using `pandas`, `numpy` or other libraries

    import pandas as pd
    data = pd.read_csv("my_data.csv")

After that, separate them into feature vectors and their corresponding 
classes

    y = data["class"].values
    X = data.drop("class").values
    
Ensure that feature vectors (X) and their classes (y) are a numpy 
matrix and a numpy array respectively. (In the example we use the
property `DataFrame.values` to convert pandas dataframe to numpy 
elements). And also ensure that classes are integers from `0` to
`num_classes - 1`. Strings are not permitted as class values.


### Create the model

    DSC = DSClassifierMultiQ(3, max_iter=150, debug_mode=True, 
                        lossfn="MSE", min_dloss=0.0001, lr=0.005,
                        precompute_rules=True)

In this step we create the model and set the configuration, the only
required parameter is the first which indicates the number of classes
in the problem (3 in our case). The rest of the parameters are optional
and are the following:

- `lr` : Initial learning rate
- `min_iter` : Minimum number of epochs to perform in the training phase
- `max_iter` : Maximun number of epochs to perform in the training phase
- `min_dloss` : Minium variation of loss to consider convergence
- `optim` : ( adam | sgd ) Optimization Method
- `lossfn` : ( CE | MSE ) Loss function
- `debug_mode` : Enables debug in training (prints and outputs metrics)
- `batch_size` : For large datasets, the number of records to be 
processed together (batch)
- `precompute_rules` : Whether to store the result of the rules 
computations for each record instead of computing every time. 
It speeds up the training but requires more memory.

### Rule definition

After the model is defined we need to define the rules. There are 2 ways
to define rules manually and automatically.

#### Define a rule manually

    from ds.DSRule import DSRule
    DSC.model.add_rule(DSRule(lambda x: x[0] > 18, "Patient is adult"))

In this case we use the method `add_rule` from our defined model. This
method accept a `DSRule` as argument. A `DSRule` can be defined directly
using its constructor which requires as first argument a *lambda* 
function which given a feature vector `x` it must return whther the rule
is satisfied (a boolean `True` or `False`). The second argument is 
optional and provides a meaningful description of the rule. In the 
exmaple if the first column of the feature vector indicates the age of a
patient, the lambda `x : x[0] > 18` is satisfied when the patient is adult,
which match the description given as second argument.

#### Define rules automatically

The model provides methods to generate rules automatically based on 
given parameters and statistics. The main two methods to generate rules
are explained below.

    DSC.model.generate_statistic_single_rules(X, breaks=3, 
                             column_names=names)
                             
Given a sample of feature vectors (usually the same using for training)
and a number of breaks `n`, the model generate simple one-attribute
rules that separate each variable into `n+1` equal-number groups. Columns
names are optional and only used to generate the descriptions.

    DSC.model.generate_mult_pair_rules(X, column_names=names)
                             
Given a sample of feature vectors (usually the same using for training).
It creates a rule for each pair of attributes indicating whether they 
are both below their means, above their means, or one above and the 
other below.

### Training

    DSC.fit(X,y)
   
The method `fit` given a set of feature vectors `X` and their 
corresponding classes `y`,  performs all the training of the model
according to the configuration and the rules defined. When this method
finishes, the model is trained so that it can predict new instances 
as accurate as possible.

Training process performs a lot of computations, therefore this method
could take several minutes to finish.

When `debug_mode` is `True` this method can also print its progress 
(e.g. the loss in each iteration) and it also measures and outputs the 
time taken in every step.

### Predicting

    y_pred = DSC.predict(X_new)

For predicting a set of new feature vectors `X_new`, the model provides
the method `predict` which returns an array with the predicted classes
for each feature vector (in the same notation as used in the fit method). 

    y_score = DSC.predict_proba(X_new)

The model also provides the method `predict_proba` which instead of 
returninga single value for each feature vector, it reaturn the 
estimated probability of belonging to each class.

### Interpretability

    DSC.model.print_most_important_rules()

The model can explain the decision it made. After training the model can
show which of the defined rules are the most important for the prediction
of each class. The method `print_most_important_rules` prints a summary
if this findings, and the method `find_most_important_rules` returns this
information in a structured way.

### Save and Load trained models

As explained before, training is a very costly operation. Then, it is not 
desirable to train the model every single time we perform a new experiment
if we already have trained it. To handle this, the model provides methods 
to save and load trained models from disk.

    DSC.model.save_rules_bin("my_trained_model.dsb")
    # ...
    DSC.model.load_rules_bin("my_trained_model.dsb")
    
Currently the model only saves the rules (lambdas and adjusted values). 
However, the other configurations must be set every time. Note that the 
model is created when invoking to `load_rules_bin` so we have already
defined its configuration.


### Full example

For a full and simple example please refer to the [Iris example](https://github.com/Sergio-P/DSGD/blob/master/examples/ds_model_iris_3.py). 
Uncomment and comment lines to see other features. 

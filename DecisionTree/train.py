import numpy as np
from sklearn import datasets, model_selection

from random_forest import RandomForestClassfier
from decisionTree import DecisionTree

data = datasets.load_iris()

x = np.array(data.data)
y = np.array(data.target)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.25, random_state=0
)
print("Train shape for iris:", x_train.shape)
print("Test Shape for iris:", x_test.shape)

print("#########################IRIS######################")

print("-------------DecisionTree----------------")

my_tree = DecisionTree(max_depth=4, min_samples_leaf=1)
my_tree.train(x_train, y_train)

train_preds = my_tree.predict(x_set=x_train)
print("Train Performance")
print(
    "Train size",
    len(y_train),
    "True preds",
    sum(train_preds == y_train),
    "Train accuracy",
    sum(train_preds == y_train) / len(y_train),
)

test_preds = my_tree.predict(x_set=x_test)
print("Test Performance")
print(
    "Test size",
    len(y_test),
    "True preds",
    sum(test_preds == y_test),
    "Test accuracy",
    sum(test_preds == y_test) / len(y_test),
)


print("----------------RF---------------")

my_tree = RandomForestClassfier(
    n_base_learner=50,
    numb_of_features_splitting="sqrt",
)
my_tree.train(x_train, y_train)


train_preds = my_tree.predict(x_set=x_train)
print("Train Performance")
print(
    "Train size",
    len(y_train),
    "True preds",
    sum(train_preds == y_train),
    "Train accuracy",
    sum(train_preds == y_train) / len(y_train),
)

test_preds = my_tree.predict(x_set=x_test)
print("Test Performance")
print(
    "Test size",
    len(y_test),
    "True preds",
    sum(test_preds == y_test),
    "Test accuracy",
    sum(test_preds == y_test) / len(y_test),
)


print("######################Breast#############################")

data = datasets.load_breast_cancer()

x = np.array(data.data)
y = np.array(data.target)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=0
)
print("Train shape for breast:", x_train.shape)
print("Test Shape for breast:", x_test.shape)

print("-------------DecisionTree----------------")

my_tree = DecisionTree(max_depth=4, min_samples_leaf=5, min_infromation_gain=0.05)
my_tree.train(x_train, y_train)

train_preds = my_tree.predict(x_set=x_train)
print("Train Performance")
print(
    "Train size",
    len(y_train),
    "True preds",
    sum(train_preds == y_train),
    "Train accuracy",
    sum(train_preds == y_train) / len(y_train),
)

test_preds = my_tree.predict(x_set=x_test)
print("Test Performance")
print(
    "Test size",
    len(y_test),
    "True preds",
    sum(test_preds == y_test),
    "Test accuracy",
    sum(test_preds == y_test) / len(y_test),
)


print("----------------RF---------------")

my_tree = RandomForestClassfier(
    n_base_learner=100, max_depth=4, min_samples_leaf=5, min_information_gain=0.05
)
my_tree.train(x_train, y_train)


train_preds = my_tree.predict(x_set=x_train)
print("Train Performance")
print(
    "Train size",
    len(y_train),
    "True preds",
    sum(train_preds == y_train),
    "Train accuracy",
    sum(train_preds == y_train) / len(y_train),
)

test_preds = my_tree.predict(x_set=x_test)
print("Test Performance")
print(
    "Test size",
    len(y_test),
    "True preds",
    sum(test_preds == y_test),
    "Test accuracy",
    sum(test_preds == y_test) / len(y_test),
)

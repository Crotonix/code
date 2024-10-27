import numpy as np
from sklearn import datasets, model_selection

from decisionTree import DecisionTree

# iris = datasets.load_iris()
data = datasets.load_breast_cancer()

x = np.array(data.data)
y = np.array(data.target)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=0
)
print("Train shape:", x_train.shape)
print("Test Shape:", x_test.shape)


my_tree = DecisionTree(max_depth=4, min_samples_leaf=1)
my_tree.train(x_train, y_train)
my_tree.print_tree()


train_preds = my_tree.predict(x_set=x_train)
print("Train Performance")
print("Train size", len(y_train))
print("True preds", sum(train_preds == y_train))
print("Train accuracy", sum(train_preds == y_train) / len(y_train))

test_preds = my_tree.predict(x_set=x_test)
print("Test Performance")
print("Test size", len(y_test))
print("True preds", sum(test_preds == y_test))
print("Test accuracy", sum(test_preds == y_test) / len(y_test))

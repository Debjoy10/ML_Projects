from dataset import Mall_Dataset
import pandas as pd 
import numpy as np

# Get Dataset Objects
train_dataset = Mall_Dataset('train')
dev_dataset = Mall_Dataset('val')

# Get data from dataset
X_train, y_train = train_dataset.get_dataset()
X_dev, y_dev = dev_dataset.get_dataset()

# Initialise Decision Tree Classifier
from decision_tree import DecisionTreeClassifier
DT_Classifier = DecisionTreeClassifier(max_depth = 4)

# Train
DT_Classifier.fit(X_train, y_train)

# Do Inference
train_acc, y_pred_train = DT_Classifier.predict(X_train, y_train)
dev_acc, y_pred_dev = DT_Classifier.predict(X_dev, y_dev)

print("Decision Tree from Scratch")
print("Train Accuracy: {}".format(train_acc))
print("Dev Accuracy: {}".format(dev_acc))

# Visualise tree
DT_Classifier.visualize()

# Checking using SkLearn Decision Tree implementation
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

# Data as one-hot encodings; since SKlearn does not provide support for categorical features
X_train, y_train = train_dataset.get_dataset(sklearn_compatible = True)
X_dev, y_dev = dev_dataset.get_dataset(sklearn_compatible = True)

# Train Classifier
clf = DecisionTreeClassifier(random_state=0, max_depth = 4)
clf.fit(X_train, y_train)

# Get Predictions
y_train_preds = clf.predict(X_train)
y_dev_preds = clf.predict(X_dev)
print("Decision Tree Sk-Learn")
print("Train Accuracy: {}".format(accuracy_score(y_train, y_train_preds)))
print("Dev Accuracy: {}".format(accuracy_score(y_dev, y_dev_preds)))

# Visualize Tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf, feature_names = X_train.columns, filled = True)
fig.savefig('Decision_Tree_Sklearn.png')
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

print("Train Accuracy: {}".format(train_acc))
print("Dev Accuracy: {}".format(dev_acc))

# Visualise tree
DT_Classifier.visualize()
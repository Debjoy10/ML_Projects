# Imports
from dataset import Mall_Dataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier

# Plotting Utility function
def plot_acc(depths, train_acc, dev_acc, fname = 'Xval.png'):
	x = list(depths)
	plt.xlabel("Depth")
	plt.ylabel("Accuracy")
	plt.title("Cross-Val training performance")
	plt.plot(x,train_acc,label = 'train')
	plt.plot(x,dev_acc,label = 'dev')
	plt.legend()
	plt.savefig(fname)

# CROSS-VALIDATION
# Parameters
k = 10
max_depth = 20

# K-Fold Cross-Validation
from sklearn.model_selection import KFold
kfold = KFold(k)
depths = np.arange(1, max_depth, 1) 

# Get Dataset Objects
train_dataset = Mall_Dataset('train')
dev_dataset = Mall_Dataset('val')

# Get data from dataset
X_train, y_train = train_dataset.get_dataset()
X_dev, y_dev = dev_dataset.get_dataset()

# Saving accuracy at each depth
train_acc_CV = []
dev_acc_CV = []

for depth in depths:
	# Cross-validation performances at current depth
	train_acc_iter = []
	dev_acc_iter = []
	print("Depth = {}".format(depth))
	idx = 0
	
	for (train, test) in kfold.split(X_train):
		# Select data subset
		X_train_CV = X_train.iloc[train, :]
		X_dev_CV = X_train.iloc[test, :]
		y_train_CV = y_train.iloc[train]
		y_dev_CV = y_train.iloc[test]
		
		# Train model with appropriate max_depth
		DT_Classifier_CV = DecisionTreeClassifier(max_depth = depth)
		DT_Classifier_CV.fit(X_train_CV, y_train_CV)

		# Inference and store
		train_acc, y_pred_train = DT_Classifier_CV.predict(X_train_CV, y_train_CV)
		dev_acc, y_pred_dev = DT_Classifier_CV.predict(X_dev_CV, y_dev_CV)	
		train_acc_iter.append(train_acc)
		dev_acc_iter.append(dev_acc)
		idx += 1
		print("Cross-Val Iter: {}, Cross-Val Train Accuracy: {}, Cross-Val Dev Accuracy: {}".format(idx, train_acc, dev_acc))

	# Collect all metrics
	train_acc_CV.append(np.mean(train_acc_iter))
	dev_acc_CV.append(np.mean(dev_acc_iter))

# Plot Performance Plots
plot_acc(depths = depths, train_acc = train_acc_CV, dev_acc = dev_acc_CV, fname = 'plots/Xval.png')

# Find optimal depth
optimal_depth = depths[np.argmax(dev_acc_CV)]
print("Optimal Depth Found = {}".format(optimal_depth))

# Train with optimal depth
DT_Classifier_best_CV = DecisionTreeClassifier(max_depth = optimal_depth)
DT_Classifier_best_CV.fit(X_train, y_train)

# Do Inference
train_acc, y_pred_train = DT_Classifier_best_CV.predict(X_train, y_train)
dev_acc, y_pred_dev = DT_Classifier_best_CV.predict(X_dev, y_dev)

print("Decision Tree having optimal depth:")
print("Train Accuracy: {}".format(train_acc))
print("Dev Accuracy: {}".format(dev_acc))

# Visualise tree
DT_Classifier_best_CV.visualize('plots/Best_Decision_Tree_Visualization.png')
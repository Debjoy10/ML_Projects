# Imports
from dataset import Mall_Dataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier
from pprint import pprint

# Plotting Utility function
def plot_acc(depths, train_acc, dev_acc, title, fname = 'Xval.png'):
	x = list(depths)
	plt.xlabel("Depth of the Decision Tree")
	plt.ylabel("Accuracy")
	plt.title("Cross-Val training performance at different depths {}".format(title))
	plt.plot(x,train_acc,label = 'train')
	plt.plot(x,dev_acc,label = 'dev')
	plt.legend()
	plt.savefig(fname)

# CROSS-VALIDATION
# Parameters
k = 10
max_depth = 20

# K-Fold Cross-Validation at different depths
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
train_acc_CV_gini = []
dev_acc_CV_gini = []

train_acc_CV_entropy = []
dev_acc_CV_entropy = []

# K Fold Cross Validation on Own Model at Different Depths

for depth in depths:
	# Cross-validation performances at current depth
	train_acc_iter_gini = []
	dev_acc_iter_gini = []

	train_acc_iter_entropy = []
	dev_acc_iter_entropy = []

	print("Depth = {}".format(depth))
	idx = 0
	
	for (train, test) in kfold.split(X_train):
		# Select data subset
		X_train_CV = X_train.iloc[train, :]
		X_dev_CV = X_train.iloc[test, :]
		y_train_CV = y_train.iloc[train]
		y_dev_CV = y_train.iloc[test]
		
		# Train model with appropriate max_depth
		DT_Classifier_CV_gini = DecisionTreeClassifier(max_depth = depth, info_metric = 'Gini')
		DT_Classifier_CV_entropy = DecisionTreeClassifier(max_depth = depth, info_metric = 'Entropy')

		DT_Classifier_CV_gini.fit(X_train_CV, y_train_CV)
		DT_Classifier_CV_entropy.fit(X_train_CV,y_train_CV)

		# Inference and store
		train_acc_gini, y_pred_train_gini = DT_Classifier_CV_gini.predict("train", X_train_CV, y_train_CV)
		train_acc_entropy, y_pred_train_entropy = DT_Classifier_CV_entropy.predict("dev", X_train_CV, y_train_CV)

		dev_acc_gini, y_pred_dev_gini = DT_Classifier_CV_gini.predict("dev", X_dev_CV, y_dev_CV)	
		dev_acc_entropy, y_pred_dev_entropy = DT_Classifier_CV_entropy.predict("dev", X_dev_CV, y_dev_CV)

		train_acc_iter_gini.append(train_acc_gini)
		dev_acc_iter_gini.append(dev_acc_gini)

		train_acc_iter_entropy.append(train_acc_entropy)
		dev_acc_iter_entropy.append(dev_acc_entropy)

		idx += 1
		print("Gini Cross-Val Iter: {}, Cross-Val Train Accuracy: {}, Cross-Val Dev Accuracy: {}".format(idx, train_acc_gini, dev_acc_gini))
		print("Entropy Cross-Val Iter: {}, Cross-Val Train Accuracy: {}, Cross-Val Dev Accuracy: {}".format(idx, train_acc_entropy, dev_acc_entropy))

	# Collect all metrics
	train_acc_CV_gini.append(np.mean(train_acc_iter_gini))
	dev_acc_CV_gini.append(np.mean(dev_acc_iter_gini))

	train_acc_CV_entropy.append(np.mean(train_acc_iter_entropy))
	dev_acc_CV_entropy.append(np.mean(dev_acc_iter_entropy))

# Plot Performance Plots
plot_acc(depths = depths, train_acc = train_acc_CV_gini, dev_acc = dev_acc_CV_gini, title = 'gini', fname = 'plots/Xval_Gini.png')
plot_acc(depths = depths, train_acc = train_acc_CV_entropy, dev_acc = dev_acc_CV_entropy, title = 'entropy', fname = 'plots/Xval_entropy.png')

np.savetxt("preds/gini_accs_t.csv", train_acc_CV_gini, delimiter=",")
np.savetxt("preds/ID3_accs_t.csv", train_acc_CV_entropy, delimiter=",")

np.savetxt("preds/gini_accs_d.csv", dev_acc_CV_gini, delimiter=",")
np.savetxt("preds/ID3_accs_d.csv", dev_acc_CV_entropy, delimiter=",")
# # Find optimal depth
# optimal_depth = depths[np.argmax(dev_acc_CV)]
# print("Optimal Depth Found = {}".format(optimal_depth))

# DT_Classifier_best_CV = DecisionTreeClassifier(max_depth = optimal_depth)
# DT_Classifier_best_CV.fit(X_train, y_train)

# # Do Inference
# train_acc, y_pred_train = DT_Classifier_best_CV.predict("train", X_train, y_train)
# dev_acc, y_pred_dev = DT_Classifier_best_CV.predict("dev", X_dev, y_dev)

# print("Decision Tree having optimal depth:")
# print("Train Accuracy: {}".format(train_acc))
# print("Dev Accuracy: {}".format(dev_acc))

# # Visualise tree
# DT_Classifier_best_CV.visualize('plots/Best_Decision_Tree_Visualization.png')

# Variation of Accuracy at each run of the K-Fold Cross Validation for a fixed depth

# For Pruned Tree

# train_acc_iter = []
# dev_acc_iter = []

# Set Depth to 7 to get the variation for the optimal depth
# depth = 4
# print("Depth = {}".format(depth))
# idx = 0
# range_i = np.arange(1,10.01,1)

# for (train, test) in kfold.split(X_train):
# 	# Select data subset
# 	X_train_CV = X_train.iloc[train, :]
# 	X_dev_CV = X_train.iloc[test, :]
# 	y_train_CV = y_train.iloc[train]
# 	y_dev_CV = y_train.iloc[test]
	
# 	# Train model with appropriate depth
# 	DT_Classifier_CV = DecisionTreeClassifier(max_depth = depth)
# 	DT_Classifier_CV.fit(X_train_CV, y_train_CV)

# 	# Inference and store
# 	train_acc, y_pred_train = DT_Classifier_CV.predict("train", X_train_CV, y_train_CV)
# 	dev_acc, y_pred_dev = DT_Classifier_CV.predict("dev", X_dev_CV, y_dev_CV)	
# 	train_acc_iter.append(train_acc)
# 	dev_acc_iter.append(dev_acc)
# 	idx += 1
# 	print("Cross-Val Iter: {}, Cross-Val Train Accuracy: {}, Cross-Val Dev Accuracy: {}".format(idx, train_acc, dev_acc))

# Plot the variation of accuracy
# plot_acc(depths = range_i, train_acc = train_acc_iter, dev_acc = dev_acc_iter, fname = 'plots/Xval_4.png')
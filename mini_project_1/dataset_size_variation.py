# Imports
from dataset import Mall_Dataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier

# Get Dataset Objects
train_dataset = Mall_Dataset('train')
dev_dataset = Mall_Dataset('val')

# Get data from dataset
X_train, y_train = train_dataset.get_dataset()
X_dev, y_dev = dev_dataset.get_dataset()

# PERFORMANCE VARIATION WITH DATASET SIZE
# We check with 5%, 10% ... 100% of training data at optimal depth

# Plotting Utility function
def plot_acc_size_var(sizes, train_acc, dev_acc, fname):
	x = [100*i for i in list(sizes)]
	plt.xlabel("Percentage of training data")
	plt.ylabel("Accuracy")
	plt.title("Validation performance with dataset size")
	plt.plot(x,train_acc,label = 'Optimal Depth N = 7')
	plt.plot(x,dev_acc,label = 'Pruned Depth N = 4')
	plt.legend()
	plt.savefig(fname)

train_accs_min = []
dev_accs_min = []

train_accs_opt = []
dev_accs_opt = []

optimal_depth = 7
min_depth = 4

# All percentages of data size
sizes = np.arange(0.1, 1.01, 0.1)

for perc in sizes:
	train_acc_iter = []
	dev_acc_iter = []
	print("Percentage of training data = {}%".format(perc*100))
	idx = 0

	# Select data subset
	X_train_sub= X_train.iloc[:int(perc*len(X_train)), :]
	y_train_sub = y_train.iloc[:int(perc*len(X_train))]
		
	# Train with optimal depth
	DT_Classifier1 = DecisionTreeClassifier(max_depth = optimal_depth)
	DT_Classifier2 = DecisionTreeClassifier(max_depth = min_depth)
	DT_Classifier1.fit(X_train_sub, y_train_sub)
	DT_Classifier2.fit(X_train_sub, y_train_sub)
	# Do Inference
	train_acc_1, y_pred_train_1 = DT_Classifier1.predict(X_train_sub, y_train_sub)
	dev_acc_1, y_pred_dev_1 = DT_Classifier1.predict(X_dev, y_dev)

	train_acc_2, y_pred_train_2 = DT_Classifier2.predict(X_train_sub, y_train_sub)
	dev_acc_2, y_pred_dev_2 = DT_Classifier2.predict(X_dev, y_dev)

	# Collect all metrics
	# print("Train Accuracy: {}".format(train_acc))
	# print("Dev Accuracy: {}".format(dev_acc))
	train_accs_opt.append(train_acc_1)
	dev_accs_opt.append(dev_acc_1)

	train_accs_min.append(train_acc_2)
	dev_accs_min.append(dev_acc_2)

# Plot Performance Plots
plot_acc_size_var(sizes = sizes, train_acc = dev_accs_opt, dev_acc = dev_accs_min, fname = 'plots/DevDataset_size_variation.png')
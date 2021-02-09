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
	plt.title("Training performance with dataset size")
	plt.plot(x,train_acc,label = 'train')
	plt.plot(x,dev_acc,label = 'dev')
	plt.legend()
	plt.savefig(fname)

train_accs = []
dev_accs = []
optimal_depth = 7

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
	DT_Classifier = DecisionTreeClassifier(max_depth = optimal_depth)
	DT_Classifier.fit(X_train_sub, y_train_sub)

	# Do Inference
	train_acc, y_pred_train = DT_Classifier.predict(X_train_sub, y_train_sub)
	dev_acc, y_pred_dev = DT_Classifier.predict(X_dev, y_dev)

	# Collect all metrics
	print("Train Accuracy: {}".format(train_acc))
	print("Dev Accuracy: {}".format(dev_acc))
	train_accs.append(train_acc)
	dev_accs.append(dev_acc)

# Plot Performance Plots
plot_acc_size_var(sizes = sizes, train_acc = train_accs, dev_acc = dev_accs, fname = 'plots/Dataset_size_variation.png')
from dataset import Mall_Dataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def plot_acc_size_var(sizes, acc_1, acc_2, fname):
	x = [i for i in list(sizes)]
	plt.xlabel("Maximum Depth of Decision Tree Allowed")
	plt.ylabel("Accuracy")
	plt.title("Comparison of Validation Data performance with max Tree Depth")
	plt.plot(x,acc_1,label = 'Model Built from scratch')
	plt.plot(x,acc_2,label = 'Scikit Pre built implementation')
	plt.legend()
	plt.savefig(fname)

# Get Dataset Objects
train_dataset = Mall_Dataset('train')
dev_dataset = Mall_Dataset('val')
test_dataset = Mall_Dataset('test')

# Get data from dataset
X_train, y_train = train_dataset.get_dataset()
X_dev, y_dev = dev_dataset.get_dataset()
X_test, y_test = test_dataset.get_dataset()

# Initialise Decision Tree Classifier
from decision_tree import DecisionTreeClassifier as DTC1


train_accs_gini_list = []
dev_accs_gini_list = []
test_accs_gini_list = []

train_accs_ID3_list = []
dev_accs_ID3_list = []
test_accs_ID3_list = []

train_accs_scikit = []
dev_accs_scikit = []

depth = np.arange(1, 25.01, 1)

# for i in depth:
# 	DT_Classifier_gini = DecisionTreeClassifier(max_depth = i, info_metric = 'gini')
# 	DT_Classifier_ID3 = DecisionTreeClassifier(max_depth = i, info_metric = 'entropy')

# 	# Train
# 	DT_Classifier_gini.fit(X_train, y_train)
# 	DT_Classifier_ID3.fit(X_train, y_train)

# 	# Do Inference
# 	train_acc_ID3, y_pred_train_ID3 = DT_Classifier_ID3.predict(X_train, y_train)
# 	dev_acc_ID3, y_pred_dev_ID3 = DT_Classifier_ID3.predict(X_dev, y_dev)
# 	#test_acc_ID3, y_pred_test_ID3 = DT_Classifier_ID3.predict(X_test, y_test)

# 	#print(y_pred_dev_ID3)

# 	train_acc_gini, y_pred_train_gini = DT_Classifier_gini.predict(X_train, y_train)
# 	dev_acc_gini, y_pred_dev_gini = DT_Classifier_gini.predict(X_dev, y_dev)
# 	#test_acc_gini, y_pred_test_gini = DT_Classifier_gini.predict(X_test, y_test)

# 	train_accs_gini_list.append(train_acc_gini);
# 	dev_accs_gini_list.append(dev_acc_gini);
# 	#test_accs_gini_list.append(test_acc_gini);

# 	train_accs_ID3_list.append(train_acc_ID3);
# 	dev_accs_ID3_list.append(dev_acc_ID3);
# 	#test_accs_ID3_list.append(test_acc_ID3);

# plot_acc_size_var(sizes = depth, acc_1 = dev_accs_gini_list, acc_2 = dev_accs_ID3_list, fname = 'plots/Gini_ID3_DevVariation.png')

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score


X_train_sck, y_train_sck = train_dataset.get_dataset(sklearn_compatible = True)
X_dev_sck, y_dev_sck = dev_dataset.get_dataset(sklearn_compatible = True)
X_test_sck, y_test_sck = test_dataset.get_dataset(sklearn_compatible = True) 

# for i in depth:
# 	DT_Classifier_gini = DTC1(max_depth = i, info_metric = 'gini')
# 	DT_Classifier_gini.fit(X_train, y_train)

# 	train_acc_gini, y_pred_train_gini = DT_Classifier_gini.predict(X_train, y_train)
# 	dev_acc_gini, y_pred_dev_gini = DT_Classifier_gini.predict(X_dev, y_dev)

# 	train_accs_gini_list.append(train_acc_gini);
# 	dev_accs_gini_list.append(dev_acc_gini);

# 	clf = DecisionTreeClassifier(max_depth = i)
# 	clf.fit(X_train_sck, y_train_sck)

# 	# Get Predictions
# 	y_train_preds = clf.predict(X_train_sck)
# 	y_dev_preds = clf.predict(X_dev_sck)
# 	train_accs_scikit.append(accuracy_score(y_train_sck,y_train_preds))
# 	dev_accs_scikit.append(accuracy_score(y_dev_sck, y_dev_preds))

DT_Classifier = DTC1(max_depth = 4, info_metric = 'gini')
DT_Classifier.fit(X_train, y_train)

clf = DecisionTreeClassifier(max_depth = 7)
clf.fit(X_train_sck, y_train_sck)

y_test_preds_scikit = clf.predict(X_test_sck)
acc, y_test_preds_own = DT_Classifier.predict(X_test)

print(y_test_preds_own)
print(y_test_preds_scikit)

np.savetxt("preds/own_depth_4.csv", y_test_preds_own, delimiter=",")
np.savetxt("preds/scikit_depth_7.csv", y_test_preds_scikit, delimiter=",")
#plot_acc_size_var(sizes = depth, acc_1 = dev_accs_gini_list, acc_2 = dev_accs_scikit, fname = 'plots/Scikit_comp_DevVariation.png')

#plot_acc_size_var(sizes = sizes, train_acc = train_accs, dev_acc = dev_accs, fname = 'plots/Dataset_size_variation.png')
#plot_acc_size_var(sizes = sizes, train_acc = train_accs, dev_acc = dev_accs, fname = 'plots/Dataset_size_variation.png')

# print("Decision Tree from Scratch")
# print("Train Accuracy: {}".format(train_acc))
# print("Dev Accuracy: {}".format(dev_acc))

# # Visualise tree
# DT_Classifier.visualize()
# print("-"*30)

# # Checking using SkLearn Decision Tree implementation
# from matplotlib import pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# from sklearn.metrics import accuracy_score

# # Data as one-hot encodings; since SKlearn does not provide support for categorical features
# X_train, y_train = train_dataset.get_dataset(sklearn_compatible = True)
# X_dev, y_dev = dev_dataset.get_dataset(sklearn_compatible = True)

# # Train Classifier
# clf = DecisionTreeClassifier(max_depth = 4)
# clf.fit(X_train, y_train)

# # Get Predictions
# y_train_preds = clf.predict(X_train)
# y_dev_preds = clf.predict(X_dev)
# print("Decision Tree Sk-Learn")
# print("Train Accuracy: {}".format(accuracy_score(y_train, y_train_preds)))
# print("Dev Accuracy: {}".format(accuracy_score(y_dev, y_dev_preds)))

# # Visualize Tree
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
# tree.plot_tree(clf, feature_names = X_train.columns, filled = True)
# PATH = 'plots/Decision_Tree_Sklearn.png'
# fig.savefig(PATH)
# print("SKLearn Decision Tree diagram written to {}".format(PATH))
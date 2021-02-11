import pandas as pd
from numpy import genfromtxt 
import numpy as np
import matplotlib.pyplot as plt

def plot_acc(depths, train_acc, dev_acc, title, fname = 'Xval.png'):
	x = list(depths)
	plt.xlabel("Depth of the Decision Tree")
	plt.ylabel("Accuracy")
	plt.title("Cross-Val training performance at different depths {}".format(title))
	plt.plot(x,train_acc,label = 'train')
	plt.plot(x,dev_acc,label = 'dev')
	plt.legend()
	plt.savefig(fname)

gini_t = genfromtxt('preds/gini_accs_t.csv', delimiter=',')
gini_d = genfromtxt('preds/gini_accs_d.csv', delimiter=',')
depths = np.arange(1, 20, 1) 
plot_acc(depths, gini_t, gini_d, 'Gini', 'plots/gini_final.png')
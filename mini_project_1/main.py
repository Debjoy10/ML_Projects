from dataset import Mall_Dataset
import pandas as pd 
import numpy as np
ds = Mall_Dataset('train')
X, y = ds.get_dataset()
from decision_tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(X, y)
print(dt.best_split(X, y))
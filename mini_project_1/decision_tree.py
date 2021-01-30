from dataset import Mall_Dataset
import pandas as pd
import numpy as np

class DTnode:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feat = None
        self.thr = None
        self.children = {}

class DecisionTreeClassifier:
    def __init__(self, max_depth = None, info_metric = 'gini'):
        self.max_depth = max_depth
        self.info_metric = info_metric
        self.num_classes = None
        self.n_features = None
    
    def get_info_metric(self, class_pop, m):
        if self.info_metric == 'gini':
            # Gini Impurity
            return 1 - np.sum([(n/m)**2 for n in class_pop])
        else:
            # Entropy
            return -1 * np.sum([(n/m)*np.log(n/m) for n in class_pop])

    def best_split(self, X, y):
        m = len(y)
        if m <= 1:
            return None, None
        
        # Population in each class for information gain calculation
        num_parent = [np.sum(y==k) for k in range(self.num_classes)]
        best_info = self.get_info_metric(num_parent, m)
        best_feat, best_thr = None, None

        for idx in range(self.n_features):
            if X.loc[:, X.columns[idx]].dtype == 'int64':
                # Continuous Data
                thresh, classes = zip(*sorted(zip(X.loc[:, X.columns[0]], y)))
                num_left = np.zeros(self.num_classes)
                num_right = num_parent.copy()

                for i in range(1, m):
                    c = classes[i-1]
                    num_left[c] += 1
                    num_right[c] -= 1

                    info_left = self.get_info_metric(num_left, m)
                    info_right = self.get_info_metric(num_right, m)
                    info_total = (i/m)*info_left + ((m-i)/m)*info_right
                    if thresh[i] == thresh[i - 1]:
                        continue

                    if info_total < best_info:
                        best_info = info_total
                        best_feat = idx
                        best_thr = (thresh[i]-thresh[i-1])/2
                
            else:
                # Categorical Data
                categories = list(X.loc[:, X.columns[idx]].dtype.categories)
                info_total = 0
                for cat in categories:
                    X_subset = X[X[X.columns[idx]] == cat]
                    y_subset = y[X[X.columns[idx]] == cat]
                    num_subset = [np.sum(y==k) for k in range(self.num_classes)] 
                    info_total += (len(X_subset)*self.get_info_metric(num_subset, m)/m)
                
                if info_total < best_info:
                    best_info = info_total
                    best_feat = idx
                    best_thr = None
            
        return best_feat, best_thr

    def grow_tree(self, X, y, depth = 0):
        num_samples = [np.sum(y==k) for k in range(self.num_classes)]
        predicted_class = np.argmax(num_samples)
        node = DTnode(predicted_class)

        if depth < self.max_depth:
            idx, thr = self.best_split(X, y)
            node.feat = idx
            node.thr = thr

            if idx == None: return node
            print(idx)
            print(X.columns[idx])
            print(thr)
            print("------------------")
            
            if idx != None:
                if thr != None:
                    left_idxs = X.iloc[:, idx] < thr
                    X_left = X.loc[left_idxs]
                    y_left = y[left_idxs]
                    node.children["left"] = self.grow_tree(X_left, y_left, depth + 1)
                    
                    right_idxs = X.iloc[:, idx] > thr
                    X_right = X.loc[right_idxs]
                    y_right = y[right_idxs]
                    node.children["right"] = self.grow_tree(X_right, y_right, depth + 1)
                else:
                    categories = list(X.loc[:, X.columns[idx]].dtype.categories)
                    for cat in categories:
                        X_subset = X[X[X.columns[idx]] == cat]
                        y_subset = y[X[X.columns[idx]] == cat]
                        node.children[cat] = self.grow_tree(X_subset, y_subset, depth + 1)
        return node
    
    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.n_features = len(X.columns)
        self.tree = self.grow_tree(X, y)
        return self.tree
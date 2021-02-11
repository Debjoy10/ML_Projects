from dataset import Mall_Dataset
import pandas as pd
import numpy as np
import pydot

class DTnode:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feat = None
        self.thr = None
        self.info_val = 0
        self.children = {}

class DecisionTreeClassifier:
    def __init__(self, max_depth = None, info_metric = 'gini'):
        self.max_depth = max_depth
        self.info_metric = info_metric
        self.num_classes = None
        self.n_features = None
    
    def get_info_metric(self, class_pop, m):
        # Function for Information Gain

        if self.info_metric == 'gini':
            # Gini Impurity
            info = 1 - np.sum([(n/m)**2 for n in class_pop])
            return info
        else:
            # Entropy
            info = -1 * np.sum([(n/m)*np.log(n/m) for n in class_pop if n > 0])
            return info

    def best_split(self, X, y):
        # Find the best feature for Decision tree node

        # If less than 1 data-point present, return
        m = len(y)
        if m <= 1:
            return None, None, 0
        
        # Population in each class for information gain calculation
        num_parent = [np.sum(y==k) for k in range(self.num_classes)]
        best_info = self.get_info_metric(num_parent, m)
        best_feat, best_thr = None, None

        for idx in range(self.n_features):
            if X.loc[:, X.columns[idx]].dtype in ['int64']:
                # Continuous Data
                thresh, classes = zip(*sorted(zip(X.loc[:, X.columns[idx]], y)))
                num_left = np.zeros(self.num_classes)
                num_right = num_parent.copy()

                for i in range(1, m):
                    c = classes[i-1]
                    num_left[c] += 1
                    num_right[c] -= 1

                    info_left = self.get_info_metric(num_left, i)
                    info_right = self.get_info_metric(num_right, m-i)
                    info_total = (i/m)*info_left + ((m-i)/m)*info_right
                    if thresh[i] == thresh[i - 1]:
                        continue

                    if info_total < best_info:
                        best_info = info_total
                        best_feat = idx
                        best_thr = (thresh[i]+thresh[i-1])/2

            else:
                # Categorical Data
                categories = list(X.loc[:, X.columns[idx]].dtype.categories)
                info_total = 0
                for cat in categories:
                    X_subset = X[X[X.columns[idx]] == cat]
                    y_subset = y[X[X.columns[idx]] == cat]
                    if len(X_subset) != 0:
                        num_subset = [np.sum(y_subset==k) for k in range(self.num_classes)] 
                        info_total += (len(X_subset)*self.get_info_metric(num_subset, len(X_subset))/m)
                
                if info_total < best_info:
                    best_info = info_total
                    best_feat = idx
                    best_thr = None
            
        return best_feat, best_thr, best_info

    def grow_tree(self, X, y, depth = 0):
        # Main function for running Decision tree algorithm and generating the tree structure

        num_samples = [np.sum(y==k) for k in range(self.num_classes)]
        predicted_class = np.argmax(num_samples)
        node = DTnode(predicted_class)

        if depth <= self.max_depth:
            idx, thr, info = self.best_split(X, y)
            node.feat = idx
            node.thr = thr
            node.info_val = info

            # Debugger
            # print("Current Depth: {}".format(depth))
            # print("Idx: {}".format(idx))
            # print("Thr: {}".format(thr))
            # print("Predicted Class: {}".format(predicted_class))
            # print("Info: {}".format(info))
            
            if idx == None: return node
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
        # Train Decision tree classifier on the X, y data passed as arguments

        self.num_classes = len(np.unique(y))
        self.n_features = len(X.columns)
        self.attr_list = X.columns
        self.tree = self.grow_tree(X, y)
        return self.tree

    def get_accuracy(self, y_true, y_pred):
        # Utility Function to calculate the accuracy from predicted and ground truth labels

        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)

    def predict(self, mode, X, y = None):
        # Inference using trained Decision Tree on the X, y data passed as arguments

        preds = []
        for i in range(len(X)):
            node = self.tree
            while node.children != {}:
                if node.thr != None:
                    if X.iloc[i][node.feat] > node.thr:
                        node = node.children["right"]
                    else:
                        node = node.children["left"]
                else:
                    node = node.children[X.iloc[i][node.feat]]

            preds.append(node.predicted_class)
        
        y_pred = np.array(preds)
        y_true = np.array(y)

        # Train or Dev
        # If you are doing inference on Training/Validation Set, Please comment out the below line
        # acc = self.get_accuracy(y_true, y_pred)
        if mode == "train" or mode == "dev":
            acc = self.get_accuracy(y_true, y_pred)
        # If you are doing inference on Test Set, Please uncomment the following lines
        # Test Set
        else:
            acc = 0;

        return acc, y_pred       

    def gen_dot(self, parent, ptext, graph):
        # Recursive function for generating the dot command for adding Parent-Children Edge to the graph viz.

        if parent.children == {}:
            return graph

        if parent.thr != None:
            rnode = parent.children["right"]
            lnode = parent.children["left"]
            
            if rnode.info_val == None: rnode.info_val = 'NA'
            ctext = self.attr_list[parent.feat] + " > " + str(parent.thr) + "\n" + "Info = " + str(rnode.info_val) + "\nPredicted Class = "  + str(rnode.predicted_class)
            edge = pydot.Edge(ptext, ctext)
            graph.add_edge(edge)
            graph = self.gen_dot(rnode, ctext, graph)

            if lnode.info_val == None: lnode.info_val = 'NA'
            ctext = ctext = self.attr_list[parent.feat] + " < " + str(parent.thr) + "\n" + "Info = " + str(lnode.info_val) + "\nPredicted Class = "  + str(lnode.predicted_class)
            edge = pydot.Edge(ptext, ctext)
            graph.add_edge(edge)
            graph = self.gen_dot(lnode, ctext, graph)

        else:
            for cat in parent.children:
                cnode = parent.children[cat]
                
                if cnode.info_val == None: cnode.info_val = 'NA'
                ctext = self.attr_list[parent.feat] + " = " + str(cat) + "\n" + "Info = " + str(cnode.info_val) + "\nPredicted Class = "  + str(cnode.predicted_class)
                edge = pydot.Edge(ptext, ctext)
                graph.add_edge(edge)
                graph = self.gen_dot(cnode, ctext, graph)
        
        return graph

    def visualize(self, PATH = 'plots/Decision_Tree_Visualization.png'):
        # Main Visualization function for that calls the Dot command generation function with the ROOT node.

        graph = pydot.Dot(graph_type='graph')
        node = self.tree
        node_text =  "ROOT" + "\n" + "Info = " + str(node.info_val) + "\nPredicted Class = "  + str(node.predicted_class)
        graph = self.gen_dot(node, node_text, graph)

        print("Decision Tree diagram written to {}".format(PATH))
        graph.write_png(PATH)
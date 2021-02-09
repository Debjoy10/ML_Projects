import pandas as pd
import numpy as np

# Dataset Class for Mall Dataset
class Mall_Dataset:
    def __init__(self, split = 'train'):
        self.split = split

        dataset = None
        if split == 'train':
            df = pd.read_csv('data/Mall_train.csv')
            dataset = df[:int(len(df)*0.9)]
        elif split == 'val':
            df = pd.read_csv('data/Mall_train.csv')
            dataset = df[int(len(df)*0.9):]
        else:
            df = pd.read_csv('data/Mall_test.csv')
            dataset = df

        self.dataset = dataset.copy()
        self.dataset = self.dataset.apply(lambda x:x.fillna(method='ffill'))
        self.all_vars = [x for x in list(self.dataset.columns) if x != 'Purchased']
        self.continuous_vars = ['Age', 'EstimatedSalary', 'Spending Score (1-100)']
        self.discrete_vars = [x for x in self.all_vars if x not in self.continuous_vars]

        # Convert to categorical
        for col in self.discrete_vars + ['Purchased']:
            self.dataset[col] = self.dataset[col].astype('category')
        
        # Collecting features and labels and filling in NaN values
        self.raw_X = self.dataset[self.all_vars]
        self.y = self.dataset['Purchased']
        self.X = self.raw_X.copy()
        self.X = self.X.apply(lambda x:x.fillna(x.value_counts().index[0]))

    def get_dataset(self, sklearn_compatible = False):
        if sklearn_compatible:
            # Convert to One-Hot-Encoded features for SKLearn
            X_new = self.X.copy()
            from sklearn.compose import ColumnTransformer 
            from sklearn.preprocessing import OneHotEncoder
            cat_idxs = [idx for idx in range(len(self.X.columns)) if self.X.iloc[:, idx].dtype != 'int64']
            columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), cat_idxs)], remainder='passthrough')
            X_new = pd.DataFrame(columnTransformer.fit_transform(X_new), dtype = np.str)  
            return X_new, self.y

        return self.X, self.y

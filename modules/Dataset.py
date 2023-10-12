import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Dataset:
    problem_type = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    @classmethod
    def load_data(cls, path):
        data = pd.read_csv(path)
        data = cls.shuffle_data(data)
        cls.problem_type = cls.determine_problem_type(data)
        X, y = cls.preprocess(data)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = cls.split_test_train(X, y)

    @classmethod
    def shuffle_data(cls, data):
        return data.sample(frac=1).reset_index(drop=True)

    @classmethod
    def determine_problem_type(cls, data):
        last_column = data.iloc[:, -1]
        dtype = last_column.dtype
        if dtype == 'float64' or dtype == 'int64':
            return "Classification"
        elif dtype == 'object':
            return "Classification"
        else:
            return "Unknown"

    @classmethod
    def preprocess(cls, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        if cls.problem_type == "Classification":
            one_hot_encoder = OneHotEncoder(sparse=False)
            y = one_hot_encoder.fit_transform(y.values.reshape(-1, 1))
        label_encoder = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = label_encoder.fit_transform(X[col])
        return X, y

    @classmethod
    def split_test_train(cls, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
        elif isinstance(y_train, np.ndarray):
            # No need to reset index for NumPy arrays
            pass
        else:
            raise ValueError("Unknown data type for y_train and y_test")
        
        return X_train, X_test, y_train, y_test

       
    @classmethod
    def resample_data(cls, tau, strategy='uniform'):
        if strategy == 'uniform':
            indices = np.random.choice(cls.X_train.index, min(tau, len(cls.X_train)), replace=False)
            cls.X_train = cls.X_train.loc[indices].reset_index(drop=True)
            
            if isinstance(cls.y_train, pd.DataFrame) or isinstance(cls.y_train, pd.Series):
                cls.y_train = cls.y_train.loc[indices].reset_index(drop=True)
            elif isinstance(cls.y_train, np.ndarray):
                cls.y_train = cls.y_train[indices]
            else:
                raise ValueError("Unknown data type for y_train")
                
        elif strategy == 'stratified':
            if cls.problem_type == 'Classification':
                unique_labels = np.unique(cls.y_train, axis=0)
                num_classes = len(unique_labels)
                instances_per_class = tau // num_classes
                indices = []
                
                for label in unique_labels:
                    label_indices = np.where((cls.y_train == label).all(axis=1))[0]
                    sampled_indices = np.random.choice(label_indices, instances_per_class, replace=True)
                    indices.extend(sampled_indices)
                
                cls.X_train = cls.X_train.iloc[indices].reset_index(drop=True)
                cls.y_train = cls.y_train[indices]
                
            elif cls.problem_type == 'Regression':
                num_quantiles = 4  # You can adjust this number
                instances_per_quantile = tau // num_quantiles
                quantiles = np.quantile(cls.y_train, np.linspace(0, 1, num_quantiles + 1))
                indices = []
                
                for i in range(len(quantiles) - 1):
                    lower_bound = quantiles[i]
                    upper_bound = quantiles[i + 1]
                    range_indices = np.where((cls.y_train >= lower_bound) & (cls.y_train <= upper_bound))[0]
                    
                    sampled_indices = np.random.choice(range_indices, instances_per_quantile, replace=True)
                    indices.extend(sampled_indices)
                
                cls.X_train = cls.X_train.iloc[indices].reset_index(drop=True)
                cls.y_train = cls.y_train[indices]
            else:
                raise ValueError("Unknown problem type")

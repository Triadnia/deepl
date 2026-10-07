import numpy as np
from sklearn.datasets import load_iris

def get_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    
    rng = np.random.default_rng(0)
    
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for class_idx in [0, 1, 2]:
        idx = np.where(y == class_idx)[0]
        rng.shuffle(idx)
        train_idx = idx[:35]
        test_idx = idx[35:]
        
        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])
        
        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])
        
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

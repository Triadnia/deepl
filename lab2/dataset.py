import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data(csv_path="creditcard.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path == "creditcard.csv":
        csv_path = os.path.join(script_dir, csv_path)
        
    df = pd.read_csv(csv_path)
    indices = df.index.values
    amount_raw = df['Amount'].values
    y = df['Class'].values
    X_raw = df.drop(columns=['Class', 'Time'])
    
    X_train_raw, X_temp_raw, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_raw, y, indices, test_size=0.4, stratify=y, random_state=0
    )
    
    X_val_raw, X_test_raw, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp_raw, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=0
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    return {
        'train': (X_train_scaled.astype(np.float32), y_train.astype(np.float32), idx_train),
        'val': (X_val_scaled.astype(np.float32), y_val.astype(np.float32), idx_val),
        'test': (X_test_scaled.astype(np.float32), y_test.astype(np.float32), idx_test),
        'amount_raw': amount_raw,
        'scaler': scaler
    }

if __name__ == "__main__":
    data = get_data()
    print(f"Розмір train: {data['train'][0].shape[0]}")
    print(f"Розмір val: {data['val'][0].shape[0]}")
    print(f"Розмір test: {data['test'][0].shape[0]}")
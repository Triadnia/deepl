import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data
from model import FraudDetectionModel

def main():
    parser = argparse.ArgumentParser(description="Навчання")
    parser.add_argument('--csv-path', type=str, default='creditcard.csv')
    args = parser.parse_args()

    data = get_data(args.csv_path)
    X_train, y_train, idx_train = data['train']
    X_val, y_val, idx_val = data['val']
    X_test, y_test, idx_test = data['test']
    amount_raw = data['amount_raw']
    scaler = data['scaler']

    print("--- Статистика поділу ---")
    for name, (X, y, idx) in [('Train', data['train']), ('Validation', data['val']), ('Test', data['test'])]:
        total = len(y)
        fraud = int(np.sum(y))
        print(f"{name}: {total} об'єктів, {fraud} шахрайств ({fraud/total*100:.3f}%)")

    torch.manual_seed(0)
    model = FraudDetectionModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    batch_size = 1024
    
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).view(-1, 1)

    for epoch in range(12):
        model.train()
        perm_indices = rng.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm_indices[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(X_train_t[batch_idx])
            loss = criterion(outputs, y_train_t[batch_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        s_val = torch.sigmoid(model(torch.tensor(X_val))).numpy().flatten()
        s_test = torch.sigmoid(model(torch.tensor(X_test))).numpy().flatten()

    torch.save(model.state_dict(), 'model_weights.pth')
    
    np.savez('predictions.npz', 
             idx_train=idx_train, y_train=y_train,
             idx_val=idx_val, y_val=y_val, s_val=s_val, amount_val=amount_raw[idx_val],
             idx_test=idx_test, y_test=y_test, s_test=s_test, amount_test=amount_raw[idx_test],
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    
    print("\nНавчання завершено. Артефакти збережено.")

if __name__ == "__main__":
    main()
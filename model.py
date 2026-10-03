import numpy as np

class NumPyNet:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        
        self.cache = {}
        
    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)
        
        Z2 = A1 @ self.W2 + self.b2
        
        self.cache['X'] = X
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        self.cache['Z2'] = Z2
        
        return Z2

    def compute_loss(self, logits, y):
        N = logits.shape[0]
        
        logits_max = np.max(logits, axis=1, keepdims=True)
        shifted_logits = logits - logits_max
        
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
        log_probs = shifted_logits - log_sum_exp
        
        correct_log_probs = log_probs[np.arange(N), y]
        
        loss = -np.mean(correct_log_probs)
        
        probs = np.exp(log_probs)
        self.cache['probs'] = probs
        self.cache['y'] = y
        
        return loss

    def backward(self, simulate_error=False):
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        probs = self.cache['probs']
        y = self.cache['y']
        
        N = X.shape[0]
        
        dZ2 = probs.copy()
        dZ2[np.arange(N), y] -= 1
        
        if not simulate_error:
            dZ2 /= N
            
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1.copy()
        dZ1[Z1 <= 0] = 0
        
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
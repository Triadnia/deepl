import argparse
import numpy as np
from data import get_data
from model import NumPyNet
from model_torch import PyTorchNet
from check import compare_with_pytorch, check_numerical_gradients

def main():
    parser = argparse.ArgumentParser(description="Лабораторна 1: Backpropagation")
    parser.add_argument(
        '--simulate-error', 
        action='store_true', 
        help="Тимчасово прибрати ділення на кількість об'єктів у градієнті за логітами"
    )
    args = parser.parse_args()

    X_train, y_train, _, _ = get_data()
    X, y = X_train, y_train

    rng = np.random.default_rng(0)
    
    W1 = rng.normal(0, np.sqrt(2 / 4), size=(4, 8))
    b1 = np.zeros(8)
    W2 = rng.normal(0, np.sqrt(2 / (8 + 3)), size=(8, 3))
    b2 = np.zeros(3)

    model_np = NumPyNet(W1.copy(), b1.copy(), W2.copy(), b2.copy())
    logits_np = model_np.forward(X)
    loss_np = model_np.compute_loss(logits_np, y)
    
    grads_np = model_np.backward(simulate_error=args.simulate_error)

    model_pt = PyTorchNet(W1.copy(), b1.copy(), W2.copy(), b2.copy())
    loss_pt, grads_pt = model_pt.compute_loss_and_gradients(X, y)

    if args.simulate_error:
        print("Дослід з помилкою\n")
    else:
        print("Нормальна реалізація\n")

    compare_with_pytorch(loss_np, grads_np, loss_pt, grads_pt)
    check_numerical_gradients(model_np, X, y, grads_np)

if __name__ == "__main__":
    main()

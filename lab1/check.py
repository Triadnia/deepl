import numpy as np

def compare_with_pytorch(loss_np, grads_np, loss_pt, grads_pt):

    threshold = 1e-12
    
    print("2. Звірка з PyTorch\n")
    print(f"Втрата NumPy:   {loss_np:.15f}")
    print(f"Втрата PyTorch: {loss_pt:.15f}\n")
    
    print("| Величина | Максимальна абсолютна різниця NumPy / PyTorch | Перевірку пройдено |")
    print("| :--- | :--- | :--- |")
    
    loss_diff = abs(loss_np - loss_pt)
    loss_passed = "Так" if loss_diff <= threshold else "Ні"
    print(f"| Втрата | {loss_diff:.2e} | {loss_passed} |")
    
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        grad_np = grads_np[param_name]
        grad_pt = grads_pt[param_name]
        
        max_diff = np.max(np.abs(grad_np - grad_pt))
        passed = "Так" if max_diff <= threshold else "Ні"
        
        print(f"| Градієнт {param_name} | {max_diff:.2e} | {passed} |")
    print("\n")


def check_numerical_gradients(model_np, X, y, grads_np):
    epsilon = 1e-6
    threshold = 1e-7
    
    params_to_check = [
        ('W1[0, 0]', model_np.W1, (0, 0), 'W1'),
        ('b1[0]', model_np.b1, (0,), 'b1'),
        ('W2[0, 0]', model_np.W2, (0, 0), 'W2'),
        ('b2[0]', model_np.b2, (0,), 'b2')
    ]
    
    print("3. Перевірка окремих градієнтів чисельним диференціюванням\n")
    print("| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for name, param_array, idx, grad_name in params_to_check:
        original_value = param_array[idx]
        
        param_array[idx] = original_value + epsilon
        logits_plus = model_np.forward(X)
        L_plus = model_np.compute_loss(logits_plus, y)
        
        param_array[idx] = original_value - epsilon
        logits_minus = model_np.forward(X)
        L_minus = model_np.compute_loss(logits_minus, y)
        
        param_array[idx] = original_value
        
        g_num = (L_plus - L_minus) / (2 * epsilon)
        
        g_manual = grads_np[grad_name][idx]
        
        diff = abs(g_num - g_manual)
        passed = "Так" if diff <= threshold else "Ні"
        
        print(f"| {name} | {g_manual:.8e} | {g_num:.8e} | {diff:.2e} | {passed} |")
    print("\n")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from threshold import get_candidates_fast, get_candidates_slow, find_best_threshold

def evaluate_metrics(y_true, y_scores, t):
    preds = (y_scores >= np.float32(t)).astype(int)
    y_true_int = y_true.astype(int)
    
    tp = int(np.sum((preds == 1) & (y_true_int == 1)))
    fp = int(np.sum((preds == 1) & (y_true_int == 0)))
    fn = int(np.sum((preds == 0) & (y_true_int == 1)))
    tn = int(np.sum((preds == 0) & (y_true_int == 0)))
    c = 10 * fn + fp
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    return tp, fp, fn, tn, c, prec, rec, preds

def check_small_sets():
    print("\nПеревірка реалізації на малих наборах")
    sets = {
        'A': (np.array([0, 1, 0, 1, 0, 0]), np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1])),
        'B': (np.array([0, 0, 0]), np.array([0.9, 0.6, 0.3])),
        'C': (np.array([1, 1] + [0]*10 + [0]), np.array([0.8] + [0.5]*11 + [0.2]))
    }
    
    expectations = {
        'A': {'t': 0.4, 'c': 2},
        'B': {'t': np.inf, 'c': 0},
        'C': {'t': 0.8, 'c': 10}
    }
    
    for name, (y, s) in sets.items():
        f_cands = get_candidates_fast(y, s)
        s_cands = get_candidates_slow(y, s)
        assert f_cands == s_cands, f"Розбіжність масивів у наборі {name}"
        
        best = find_best_threshold(f_cands)
        print(f"\nТаблиця кандидатів для набору {name}:")
        print(pd.DataFrame(f_cands).to_string(index=False))
        
        assert best['threshold'] == expectations[name]['t'], f"Помилка t* у {name}"
        assert best['C'] == expectations[name]['c'], f"Помилка C у {name}"

def main():
    preds = np.load('predictions.npz')
    y_val, s_val = preds['y_val'], preds['s_val']
    idx_test, y_test, s_test, amt_test = preds['idx_test'], preds['y_test'], preds['s_test'], preds['amount_test']

    check_small_sets()

    print("\nВибір порога на Validation")
    precision, recall, _ = precision_recall_curve(y_val, s_val)
    fpr, tpr, _ = roc_curve(y_val, s_val)
    auc_roc = roc_auc_score(y_val, s_val)
    auc_pr = auc(recall, precision)
    
    cands_val = get_candidates_fast(y_val, s_val)
    pd.DataFrame(cands_val).to_csv('candidates.csv', index=False)
    
    best_cand = find_best_threshold(cands_val)
    t_star = best_cand['threshold']
    
    with open('threshold.json', 'w') as f:
        json.dump({'t_star': t_star}, f)
    
    
    
    print(f"\nПравило на validation | Поріг | TP | FP | FN | TN | C")
    print("-" * 65)
    for name, t in [("Стандартний поріг", 0.5), ("Мінімум вартості (t*)", t_star), ("Усі негативні", np.inf)]:
        c = evaluate_metrics(y_val, s_val, t) # tp, fp, fn, tn, cost...
        print(f"{name:<21} | {t:<5.4f} | {c[0]:<2} | {c[1]:<2} | {c[2]:<2} | {c[3]:<5} | {c[4]}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.', markersize=2, label=f'PR (AUC={auc_pr:.4f})')
    c_05 = evaluate_metrics(y_val, s_val, 0.5)
    c_tstar = evaluate_metrics(y_val, s_val, t_star)
    plt.scatter([c_05[6]], [c_05[5]], color='red', label='t=0.5', zorder=5)
    plt.scatter([c_tstar[6]], [c_tstar[5]], color='orange', label='t*', zorder=5)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.title('PR Curve')

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.', markersize=2, label=f'ROC (AUC={auc_roc:.4f})')
    
    fpr_05 = c_05[1] / (c_05[1] + c_05[3])
    tpr_05 = c_05[0] / (c_05[0] + c_05[2])
    fpr_tstar = c_tstar[1] / (c_tstar[1] + c_tstar[3])
    tpr_tstar = c_tstar[0] / (c_tstar[0] + c_tstar[2])
    
    plt.scatter([fpr_05], [tpr_05], color='red', label='t=0.5', zorder=5)
    plt.scatter([fpr_tstar], [tpr_tstar], color='orange', label='t*', zorder=5)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('ROC Curve')
    plt.savefig('curves_val.png')

    print("\n4. Тестова оцінка та порівняння")
    print(f"{'Поріг':<10} | {'TP':<3} | {'FP':<4} | {'FN':<3} | {'TN':<5} | {'C':<5} | {'Precision':<9} | {'Recall':<9}")
    print("-" * 65)
    
    for t_name, t_val in [('t=0.5', 0.5), ('t*', t_star), ('+inf', np.inf)]:
        tp, fp, fn, tn, c, prec, rec, _ = evaluate_metrics(y_test, s_test, t_val)
        prec_str = f"{prec:.4f}" if not np.isnan(prec) else "NaN (TP+FP=0)"
        rec_str = f"{rec:.4f}" if not np.isnan(rec) else "NaN (TP+FN=0)"
        print(f"{t_name:<10} | {tp:<3} | {fp:<4} | {fn:<3} | {tn:<5} | {c:<5} | {prec_str:<9} | {rec_str:<9}")

    _, _, _, _, _, _, _, preds_test = evaluate_metrics(y_test, s_test, t_star)
    fp_mask = (preds_test == 1) & (y_test.astype(int) == 0)
    fn_mask = (preds_test == 0) & (y_test.astype(int) == 1)

    fp_rows = np.sort(idx_test[fp_mask])[:5]
    fn_rows = np.sort(idx_test[fn_mask])[:5]
    pos_map = {r: i for i, r in enumerate(idx_test)}

    errors = []
    for r in fp_rows:
        i = pos_map[r]
        errors.append({'CSV_Row': r, 'Type': 'FP', 'Label': 0, 'Score': s_test[i], 'Amount': amt_test[i], 'Dist': s_test[i] - t_star})
    for r in fn_rows:
        i = pos_map[r]
        errors.append({'CSV_Row': r, 'Type': 'FN', 'Label': 1, 'Score': s_test[i], 'Amount': amt_test[i], 'Dist': s_test[i] - t_star})

    df_errors = pd.DataFrame(errors)
    df_errors.to_csv('errors.csv', index=False)
    
    print("\nАналіз помилок (Перші 5 FP/FN за рядком CSV)")
    print(df_errors.to_string(index=False))

if __name__ == "__main__":
    main()
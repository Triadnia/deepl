import numpy as np

def get_candidates_fast(y_true, y_scores):
    P = int(np.sum(y_true))
    N = len(y_true) - P
    
    candidates = [{'threshold': np.inf, 'TP': 0, 'FP': 0, 'FN': P, 'TN': N, 'C': 10 * P}]
    
    desc_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[desc_idx]
    s_sorted = y_scores[desc_idx]
    
    distinct_idx = np.where(np.diff(s_sorted) != 0)[0]
    distinct_idx = np.concatenate([distinct_idx, [len(s_sorted) - 1]])
    
    tps_all = np.cumsum(y_sorted)
    fps_all = np.cumsum(1 - y_sorted)
    
    tps = tps_all[distinct_idx]
    fps = fps_all[distinct_idx]
    thresholds = s_sorted[distinct_idx]
    
    for i in range(len(thresholds)):
        tp, fp = int(tps[i]), int(fps[i])
        fn, tn = P - tp, N - fp
        candidates.append({
            'threshold': float(thresholds[i]),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'C': 10 * fn + fp
        })
    return candidates

def get_candidates_slow(y_true, y_scores):
    P = int(np.sum(y_true))
    N = len(y_true) - P
    unique_scores = np.unique(y_scores)[::-1]
    
    thresholds = [np.inf] + list(unique_scores)
    candidates = []
    
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        candidates.append({
            'threshold': float(t),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'C': 10 * fn + fp
        })
    return candidates

def find_best_threshold(candidates):
    min_c = min(cand['C'] for cand in candidates)
    best_cands = [c for c in candidates if c['C'] == min_c]
    return max(best_cands, key=lambda x: x['threshold'])
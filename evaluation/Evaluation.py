import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

def nll_mean_safe(y, p, eps=1e-15):
    y = np.asarray(y).astype(np.int32)
    p = np.asarray(p).astype(np.float64)
    p = np.clip(p, eps, 1.0 - eps)

    per = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.mean(per))  #mean over observations


def rce_explicit(y, p, eps=1e-15):
    y = np.asarray(y).astype(np.int32)
    p = np.asarray(p).astype(np.float64)

    p0 = float(np.clip(np.mean(y), eps, 1.0 - eps))
    p_base = np.full_like(p, p0, dtype=np.float64)

    nll_base = nll_mean_safe(y, p_base, eps=eps)
    nll_mod  = nll_mean_safe(y, p,      eps=eps)

    rce_val = (1.0 - nll_mod / max(nll_base, eps)) * 100.0
    return float(rce_val)

def topk_metrics(y_true, p_pred, *, fracs=(0.05,0.10, 0.20)):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pred).astype(float)

    n = len(y)
    if n == 0:
        return {}

    order = np.argsort(-p)  # descending
    y_sorted = y[order]

    total_pos = int(y.sum())
    base_rate = float(total_pos / n) if n > 0 else 0.0

    out = {"base_rate": base_rate}

    for f in fracs:
        k = max(1, int(round(f * n)))
        top = y_sorted[:k]
        tp = int(top.sum())

        precision = float(tp / k)  # of top-k, how many are 1
        recall = float(tp / total_pos) if total_pos else 0.0
        lift = float(precision / base_rate) if base_rate > 0 else 0.0

        key = int(round(f * 100))
        out[f"prec_{key}%"] = precision
        out[f"lift_{key}%"] = lift
        out[f"recall_{key}%"] = recall

    return out


def evaluation(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    evaluation_metrics= {}
    eps = 1e-15
    p = np.clip(p_pred.astype(np.float64), eps, 1 - eps)
    y = y_true.astype(np.int32)

    nll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    ap = average_precision_score(y, p)
    rce = rce_explicit(y, p, eps=eps)

    precision, recall, _ = precision_recall_curve(y, p)
    pr_auc = float(auc(recall, precision)) 
    
    evaluation_metrics["nll_mean"] = nll
    evaluation_metrics["rce"] = rce
    evaluation_metrics["ap"] = ap           
    evaluation_metrics["pr_auc"] = pr_auc   

    evaluation_metrics.update(topk_metrics(y, p, fracs=(0.05, 0.10, 0.20)))
    
    return evaluation_metrics
    

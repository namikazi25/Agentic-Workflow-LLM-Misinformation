from __future__ import annotations
import json, numpy as np, joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURES = ["text_support","text_refute","overlap_max","answer_conf",
            "ai_gen_prob","manip_prob","photo_like",
            "vl_support_prob","vl_refute_prob"]

def _get(d, k, default=0.0): 
    try: return float(d.get(k, default))
    except: return float(default)

def vec(rec):
    t, v, c = rec.get("head_text", {}), rec.get("head_visual", {}), rec.get("head_consistency", {})
    x = np.array([
        _get(t,"text_support",0.5),
        _get(t,"text_refute",0.5),
        _get(t,"overlap_max",0.0)/5.0,      # scale overlap
        _get(t,"answer_conf",0.5),
        _get(v,"ai_gen_prob",0.5),
        _get(v,"manip_prob",0.5),
        _get(v,"photo_like",0.5),
        _get(c,"vl_support_prob",0.5),
        _get(c,"vl_refute_prob",0.5),
    ], dtype=float)
    y = 1 if str(rec.get("label_binary","")).lower()=="fake" else 0
    return x, y

def load_jsonl(path):
    X, y = [], []
    for line in open(path, "r", encoding="utf-8"):
        r = json.loads(line)
        xi, yi = vec(r); X.append(xi); y.append(yi)
    return np.vstack(X), np.array(y, dtype=int)

def ece(y_true, p, n_bins=15):
    import numpy as np
    bins = np.linspace(0,1,n_bins+1); m = len(y_true); e=0.0
    for i in range(n_bins):
        msk = (p>=bins[i]) & (p<bins[i+1])
        if msk.sum()==0: continue
        conf=p[msk].mean(); acc=(y_true[msk]==(p[msk]>=0.5)).mean()
        e += (msk.sum()/m)*abs(acc-conf)
    return float(e)

def main(train_path, val_path, out_path="models/fuser.joblib"):
    Path("models").mkdir(parents=True, exist_ok=True)
    Xtr, ytr = load_jsonl(train_path)
    Xva, yva = load_jsonl(val_path)
    base = LogisticRegression(max_iter=500, class_weight="balanced")
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:,1]
    print("AUC:", round(roc_auc_score(yva,p),3), 
          "Brier:", round(brier_score_loss(yva,p),3), 
          "ECE:", round(ece(yva,p),3))
    joblib.dump({"clf": clf, "features": FEATURES}, out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])

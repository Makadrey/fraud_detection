"""Blend the base models.

Ensembles (all fitted on OOF predictions only, never on the test set):
  ens_mean_rank      average of percentile-ranks
  ens_weighted_rank  rank average weighted by CV PR-AUC
  ens_greedy         greedy forward selection (Caruana) maximising OOF PR-AUC
  ens_stack_lr       logistic-regression stacker on logit(probabilities)
Test scores are mapped to percentiles using the OOF score distribution, so one
threshold chosen on OOF can be applied to the test set.
Outputs: outputs/oof_ens.csv, outputs/test_ens.csv, outputs/ensemble_results.csv
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.features import META


def to_percentile(ref, x):
    """Percentile of x within the reference (OOF) score distribution."""
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def greedy_weights(R, y, rounds=20):
    chosen, best = [], -1
    for _ in range(rounds):
        scores = [ap(y, R[:, chosen + [j]].mean(axis=1)) for j in range(R.shape[1])]
        j = int(np.argmax(scores))
        if scores[j] < best - 1e-9:
            break
        chosen.append(j)
        best = max(best, scores[j])
    w = np.bincount(chosen, minlength=R.shape[1]).astype(float)
    return w / w.sum()


def main():
    oof = pd.read_csv(config.OUT / "oof_preds.csv")
    tst = pd.read_csv(config.OUT / "test_preds.csv")
    cv_res = pd.read_csv(config.OUT / "cv_results.csv").set_index("model")
    models = [c for c in oof.columns if c not in META]
    y_oof, y_tst = oof["y"].values, tst["y"].values

    R_oof = np.column_stack([to_percentile(oof[m], oof[m]) for m in models])
    R_tst = np.column_stack([to_percentile(oof[m], tst[m]) for m in models])

    out_oof, out_tst = oof[META].copy(), tst[META].copy()

    out_oof["ens_mean_rank"], out_tst["ens_mean_rank"] = R_oof.mean(1), R_tst.mean(1)

    w = cv_res.loc[models, "cv_pr_auc"].values
    w = w / w.sum()
    out_oof["ens_weighted_rank"], out_tst["ens_weighted_rank"] = R_oof @ w, R_tst @ w

    g = greedy_weights(R_oof, y_oof)
    out_oof["ens_greedy"], out_tst["ens_greedy"] = R_oof @ g, R_tst @ g

    L_oof = logit(oof[models].values)
    L_tst = logit(tst[models].values)
    stacker = LogisticRegression(C=1.0, max_iter=1000)
    cv = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
    out_oof["ens_stack_lr"] = cross_val_predict(stacker, L_oof, y_oof, cv=cv, method="predict_proba")[:, 1]
    stacker.fit(L_oof, y_oof)
    out_tst["ens_stack_lr"] = stacker.predict_proba(L_tst)[:, 1]

    ens = [c for c in out_oof.columns if c.startswith("ens_")]
    rows = [{"ensemble": e,
             "oof_pr_auc": ap(y_oof, out_oof[e]), "oof_roc_auc": roc_auc_score(y_oof, out_oof[e]),
             "test_pr_auc": ap(y_tst, out_tst[e]), "test_roc_auc": roc_auc_score(y_tst, out_tst[e])}
            for e in ens]
    res = pd.DataFrame(rows).sort_values("oof_pr_auc", ascending=False)

    out_oof.to_csv(config.OUT / "oof_ens.csv", index=False)
    out_tst.to_csv(config.OUT / "test_ens.csv", index=False)
    res.to_csv(config.OUT / "ensemble_results.csv", index=False)
    weights = {
        "greedy": dict(zip(models, g.round(3).tolist())),
        "cv_weighted": dict(zip(models, w.round(3).tolist())),
        "stack_coef": dict(zip(models, stacker.coef_[0].round(3).tolist())),
    }
    (config.OUT / "ensemble_weights.json").write_text(json.dumps(weights, indent=2))
    print(res.round(3).to_string(index=False))
    print("\ngreedy weights:", weights["greedy"])


if __name__ == "__main__":
    main()

"""Train every candidate model.

For each model:
  * 5-fold CV on the training period -> out-of-fold (OOF) predictions
  * refit on the full training period -> predictions on the held-out future test period
Outputs: outputs/oof_preds.csv, outputs/test_preds.csv, outputs/cv_results.csv, models/*.joblib
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.features import FEATURES, add_features, load_data, time_split
from src.models import get_models


def base_frame(d):
    return pd.DataFrame({"transaction_id": d["transaction_id"].values,
                         "y": d[config.TARGET].values, "amount": d["amount"].values})


def main():
    config.make_dirs()
    df = add_features(load_data())
    train, test = time_split(df)
    X_tr, y_tr = train[FEATURES], train[config.TARGET]
    X_te, y_te = test[FEATURES], test[config.TARGET]
    print(f"train {len(train):,} rows ({y_tr.mean():.2%} fraud) | "
          f"test {len(test):,} rows ({y_te.mean():.2%} fraud)\n")

    cv = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof, tst, rows = base_frame(train), base_frame(test), []

    for name, model in get_models().items():
        t0 = time.time()
        oof[name] = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        model.fit(X_tr, y_tr)
        tst[name] = model.predict_proba(X_te)[:, 1]
        joblib.dump(model, config.MODELS / f"{name}.joblib")
        rows.append({
            "model": name,
            "cv_pr_auc": average_precision_score(y_tr, oof[name]),
            "cv_roc_auc": roc_auc_score(y_tr, oof[name]),
            "test_pr_auc": average_precision_score(y_te, tst[name]),
            "test_roc_auc": roc_auc_score(y_te, tst[name]),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{name:16s} cv PR-AUC {rows[-1]['cv_pr_auc']:.3f} | "
              f"test PR-AUC {rows[-1]['test_pr_auc']:.3f} | {rows[-1]['seconds']}s")

    res = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False)
    oof.to_csv(config.OUT / "oof_preds.csv", index=False)
    tst.to_csv(config.OUT / "test_preds.csv", index=False)
    res.to_csv(config.OUT / "cv_results.csv", index=False)
    print("\n", res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

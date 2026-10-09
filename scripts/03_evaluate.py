"""Final evaluation.

* ranks every model + ensemble (choice is made on OOF, not on test)
* picks an alert threshold on OOF by minimising money lost
  (missed fraud costs the transaction amount, a false alert costs config.FP_COST)
* applies that threshold to the untouched test period
* bootstrap CI, plots, permutation importance
Outputs: outputs/final_report.csv, outputs/best_model.json, outputs/figures/*.png
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score as ap, ConfusionMatrixDisplay,
                             precision_recall_curve, roc_auc_score, roc_curve)

from src import config
from src.features import FEATURES, META, add_features, load_data, time_split


def best_cost_threshold(score, y, amount):
    """Alert on the top-k scores; choose k that minimises total cost."""
    order = np.argsort(-score)
    s, yy, aa = score[order], y[order], amount[order]
    k = np.arange(len(s) + 1)
    caught_amt = np.concatenate([[0], np.cumsum(aa * yy)])
    caught_n = np.concatenate([[0], np.cumsum(yy)])
    cost = (aa * yy).sum() - caught_amt + config.FP_COST * (k - caught_n)
    kbest = int(np.argmin(cost))
    thr = s[kbest - 1] if kbest > 0 else np.inf
    return thr, cost


def cost_of(pred, y, amount):
    fn = ((pred == 0) & (y == 1))
    fp = ((pred == 1) & (y == 0))
    return amount[fn].sum() + config.FP_COST * fp.sum()


def main():
    config.make_dirs()
    oof = pd.read_csv(config.OUT / "oof_preds.csv").merge(
        pd.read_csv(config.OUT / "oof_ens.csv").drop(columns=["y", "amount"]), on="transaction_id")
    tst = pd.read_csv(config.OUT / "test_preds.csv").merge(
        pd.read_csv(config.OUT / "test_ens.csv").drop(columns=["y", "amount"]), on="transaction_id")
    names = [c for c in oof.columns if c not in META]
    y_o, a_o = oof["y"].values, oof["amount"].values
    y_t, a_t = tst["y"].values, tst["amount"].values
    no_model_cost = a_t[y_t == 1].sum()

    rows = []
    for n in names:
        thr, _ = best_cost_threshold(oof[n].values, y_o, a_o)
        pred = (tst[n].values >= thr).astype(int)
        tp = ((pred == 1) & (y_t == 1)).sum()
        c = cost_of(pred, y_t, a_t)
        rows.append({
            "model": n,
            "oof_pr_auc": ap(y_o, oof[n]),
            "test_pr_auc": ap(y_t, tst[n]),
            "test_roc_auc": roc_auc_score(y_t, tst[n]),
            "precision": tp / max(pred.sum(), 1),
            "recall": tp / y_t.sum(),
            "alert_rate": pred.mean(),
            "test_cost": c,
            "savings_vs_no_model": 1 - c / no_model_cost,
        })
    rep = pd.DataFrame(rows).sort_values("oof_pr_auc", ascending=False).reset_index(drop=True)
    rep.to_csv(config.OUT / "final_report.csv", index=False)
    print(rep.round(3).to_string(index=False))

    best = rep.loc[0, "model"]
    thr, cost_curve = best_cost_threshold(oof[best].values, y_o, a_o)
    pred = (tst[best].values >= thr).astype(int)
    json.dump({"best_model": best, "threshold": float(thr), "fp_cost": config.FP_COST},
              open(config.OUT / "best_model.json", "w"), indent=2)
    print(f"\nbest by OOF PR-AUC: {best} | threshold {thr:.4f}")

    # bootstrap CI on test PR-AUC (best vs best single model)
    rng = np.random.default_rng(config.SEED)
    single = rep[~rep["model"].str.startswith("ens_")].iloc[0]["model"]
    idx = [rng.integers(0, len(y_t), len(y_t)) for _ in range(200)]
    for n in dict.fromkeys([best, single]):
        b = [ap(y_t[i], tst[n].values[i]) for i in idx]
        print(f"{n:18s} test PR-AUC 95% CI: [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")

    # --- plots ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for n in names:
        p, r, _ = precision_recall_curve(y_t, tst[n])
        ax[0].plot(r, p, lw=2.5 if n == best else 1, label=f"{n} ({ap(y_t, tst[n]):.2f})")
        fpr, tpr, _ = roc_curve(y_t, tst[n])
        ax[1].plot(fpr, tpr, lw=2.5 if n == best else 1)
    ax[0].axhline(y_t.mean(), ls="--", c="gray")
    ax[0].set(title="Precision-Recall (test)", xlabel="recall", ylabel="precision")
    ax[0].legend(fontsize=7)
    ax[1].plot([0, 1], [0, 1], "--", c="gray")
    ax[1].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    fig.tight_layout(); fig.savefig(config.FIG / "pr_roc_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(cost_curve)) / len(cost_curve) * 100, cost_curve)
    ax.set(title=f"OOF cost vs alert rate ({best})", xlabel="% of transactions flagged", ylabel="total cost")
    ax.set_xlim(0, 15)
    fig.tight_layout(); fig.savefig(config.FIG / "cost_curve.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_t, pred, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{best} @ cost-optimal threshold")
    fig.tight_layout(); fig.savefig(config.FIG / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rep["model"], rep["savings_vs_no_model"] * 100)
    ax.set(ylabel="% fraud loss avoided (test)", title="Savings vs no model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(config.FIG / "savings.png", dpi=130); plt.close(fig)

    # permutation importance on the strongest single model (raw features, so no encoding mess)
    model = joblib.load(config.MODELS / f"{single}.joblib")
    test_df = time_split(add_features(load_data()))[1]
    sample = test_df.sample(min(15000, len(test_df)), random_state=config.SEED)
    pi = permutation_importance(model, sample[FEATURES], sample[config.TARGET],
                                scoring="average_precision", n_repeats=3,
                                random_state=config.SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    imp.plot.barh(ax=ax)
    ax.set(title=f"Permutation importance ({single})", xlabel="drop in PR-AUC")
    fig.tight_layout(); fig.savefig(config.FIG / "permutation_importance.png", dpi=130); plt.close(fig)
    imp.sort_values(ascending=False).to_csv(config.OUT / "feature_importance.csv", header=["importance"])
    print("\nfigures saved to", config.FIG)


if __name__ == "__main__":
    main()

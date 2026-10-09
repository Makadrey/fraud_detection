# Fraud Detection: imbalanced classification, ensembling and cost-based evaluation

A small end-to-end project on transaction fraud (~2% positives): EDA, six models that each
handle class imbalance differently (class weights, SMOTE, undersampling, balanced forests),
four ensembling strategies, and an evaluation based on **money saved** rather than accuracy.

No package, just scripts and notebooks you run in order.

## Layout

```
src/
  config.py      paths, seed, split size, cost assumptions
  features.py    loading, feature engineering, time split, preprocessing
  models.py      the six candidate models (imbalanced-learn pipelines)
scripts/
  01_train_models.py   5-fold OOF predictions + test predictions for every model
  02_ensemble.py       mean-rank, weighted-rank, greedy selection, stacking
  03_evaluate.py       cost-optimal threshold, test metrics, bootstrap CI, plots
notebooks/
  01_eda.ipynb             imbalance, missingness, signals, drift, leakage checks
  02_error_analysis.ipynb  what the final model misses and where it raises false alarms
data/            put transactions.csv here (see data/README.md for the schema)
```

## Run

```bash
pip install -r requirements.txt
# place data/transactions.csv (schema in data/README.md)
jupyter notebook notebooks/01_eda.ipynb      # run from the notebooks/ folder
python scripts/01_train_models.py
python scripts/02_ensemble.py
python scripts/03_evaluate.py
jupyter notebook notebooks/02_error_analysis.ipynb
```

Results land in `outputs/` (CSV reports, `figures/`) and `models/`. Both are git-ignored.

## Design choices

- **Time-based split.** The most recent 30% of transactions is the test set. A random split would
  let the model peek at the future and overstate performance.
- **PR-AUC as the main metric.** ROC-AUC looks great even for weak fraud models when negatives dominate.
- **Out-of-fold predictions.** Every model gets 5-fold OOF predictions on the training period.
  Ensemble weights, stacker and alert threshold are all fitted on OOF only, so the test set
  is touched once, at the end.
- **Imbalance handling compared side by side.** `class_weight`, `SMOTE`, `RandomUnderSampler`
  and `BalancedRandomForestClassifier`, with `imblearn.pipeline.Pipeline` so resampling happens
  inside each CV fold and never leaks into validation data.
- **Rank-based blending.** SMOTE/undersampling distort probabilities, so averaging raw
  probabilities is misleading. Scores are converted to percentiles of the OOF distribution
  (test scores use the same mapping), plus one logistic-regression stacker on logits.
- **Cost-based threshold.** A missed fraud costs the transaction amount, a false alert costs
  `FP_COST` (analyst review, set in `src/config.py`). The threshold minimises this cost on OOF,
  then is applied unchanged to the test set. The report shows % of fraud loss avoided vs having no model.
- **Explainability.** Permutation importance on raw features of the best single model.

## Known limitations / ideas

- SMOTE interpolates one-hot and indicator columns, producing fractional values. `SMOTENC` is the stricter option.
- CV folds are random within the training period; a rolling time-series CV would be more realistic.
- Customer-level behaviour features (rolling counts, merchant novelty) would likely help most.
- No hyperparameter tuning: the point here is comparing imbalance strategies and ensembles.

"""Candidate models. Each one handles class imbalance a different way."""
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # imblearn's Pipeline: resamples during fit only
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import SEED
from .features import build_preprocessor


def _pipe(*steps):
    return Pipeline([("prep", build_preprocessor()), *steps])


def get_models() -> dict:
    return {
        # class weights
        "lr_weighted": _pipe(
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))),
        "hgb_weighted": _pipe(
            ("clf", HistGradientBoostingClassifier(
                class_weight="balanced", learning_rate=0.05, max_iter=300, random_state=SEED))),
        # oversampling
        "lr_smote": _pipe(
            ("smote", SMOTE(sampling_strategy=0.2, random_state=SEED)),
            ("clf", LogisticRegression(max_iter=1000))),
        "hgb_smote": _pipe(
            ("smote", SMOTE(sampling_strategy=0.2, random_state=SEED)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, random_state=SEED))),
        # undersampling
        "rf_undersample": _pipe(
            ("under", RandomUnderSampler(sampling_strategy=0.1, random_state=SEED)),
            ("clf", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED))),
        # undersampling inside every tree
        "balanced_rf": _pipe(
            ("clf", BalancedRandomForestClassifier(
                n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED))),
    }

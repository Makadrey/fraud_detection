"""Loading, feature engineering, time split and the preprocessing block."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

CAT = ["merchant_category", "channel", "device_type", "country"]
NUM = [
    "amount", "log_amount", "customer_age", "account_age_days", "txn_count_24h",
    "avg_amount_30d", "distance_from_home_km", "amount_to_avg_ratio",
    "hour", "is_night", "is_weekend", "is_foreign", "is_new_account",
]
FEATURES = NUM + CAT
META = ["transaction_id", "y", "amount"]  # non-model columns kept in prediction files


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["is_night"] = (df["hour"] < 6).astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_30d"] + 1)  # NaN if avg missing
    df["is_foreign"] = (df["country"] != config.HOME_COUNTRY).astype(int)
    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)
    return df


def time_split(df: pd.DataFrame):
    """Train on the past, test on the future (df must be sorted by time)."""
    cut = int(len(df) * (1 - config.TEST_FRAC))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # missingness can be a signal
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])

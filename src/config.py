"""Central settings. Change things here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
MODELS = ROOT / "models"

TARGET = "is_fraud"
SEED = 42
TEST_FRAC = 0.30      # most recent 30% of transactions = test set (time-based split)
N_FOLDS = 5           # CV folds used to build out-of-fold predictions
HOME_COUNTRY = "NG"   # anything else counts as a foreign transaction
FP_COST = 5.0         # cost of manually reviewing one false alert
# cost of a missed fraud (false negative) = the transaction amount


def make_dirs():
    for p in (OUT, FIG, MODELS):
        p.mkdir(parents=True, exist_ok=True)

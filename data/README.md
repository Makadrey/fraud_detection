# Data

The dataset is **not included** in this repo. Put your file at `data/transactions.csv`
with one row per transaction and these columns:

| column | type | notes |
|---|---|---|
| transaction_id | int/str | unique id |
| customer_id | int/str | used for EDA only, not as a feature |
| timestamp | datetime | e.g. `2026-01-31 14:05:12` |
| amount | float | transaction amount |
| merchant_category | str | e.g. grocery, electronics, crypto |
| channel | str | e.g. pos, online, atm, mobile_app |
| device_type | str | may be missing (pos/atm have no device) |
| country | str | ISO-like code; home country is set in `src/config.py` |
| customer_age | float | |
| account_age_days | float | |
| txn_count_24h | int | customer's transactions in the previous 24h |
| avg_amount_30d | float | customer's 30-day average amount, may be missing |
| distance_from_home_km | float | may be missing |
| is_fraud | 0/1 | target, expect ~1-3% positives |

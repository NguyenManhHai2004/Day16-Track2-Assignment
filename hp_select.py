import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("/home/ubuntu/ml-benchmark/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]
X_full, _, y_full, _ = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=0.25, random_state=SEED, stratify=y_full)

configs = {
    "default": {},
    "reg_lambda=1": {"reg_lambda": 1.0},
    "min_child_weight=1": {"min_child_weight": 1.0},
    "reg_lambda=5,min_child_weight=1": {"reg_lambda": 5.0, "min_child_weight": 1.0},
    "reg_lambda=1,min_child_samples=50": {"reg_lambda": 1.0, "min_child_samples": 50},
}
for name, extra in configs.items():
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, metric="auc",
                           random_state=SEED, n_jobs=1, verbose=-1, **extra)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    print(f"{name}: val_auc={auc:.4f} best_iter={m.best_iteration_}", flush=True)

import json
import statistics
import time
from datetime import datetime, timezone

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA = "/home/ubuntu/ml-benchmark/creditcard.csv"
SEED = 42
LATENCY_WARMUP, LATENCY_REPEATS = 50, 500
BATCH_WARMUP, BATCH_REPEATS = 3, 20

t0 = time.perf_counter()
df = pd.read_csv(DATA)
data_load_seconds = time.perf_counter() - t0

X = df.drop(columns=["Class"])
y = df["Class"]
X_full, X_test, y_full, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.25, random_state=SEED, stratify=y_full)

model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31, reg_lambda=1.0, min_child_samples=50,
    metric="auc", random_state=SEED, n_jobs=1, verbose=-1,
)
t0 = time.perf_counter()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
training_seconds = time.perf_counter() - t0

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

one = X_test.iloc[[0]]
for _ in range(LATENCY_WARMUP):
    model.predict(one)
t0 = time.perf_counter()
for _ in range(LATENCY_REPEATS):
    model.predict(one)
latency_1_row_ms = (time.perf_counter() - t0) / LATENCY_REPEATS * 1000

batch = X_test.iloc[:1000]
for _ in range(BATCH_WARMUP):
    model.predict(batch)
times = []
for _ in range(BATCH_REPEATS):
    t0 = time.perf_counter()
    model.predict(batch)
    times.append(time.perf_counter() - t0)
batch_seconds = statistics.median(times)

result = {
    "instance": {"shape": "VM.Standard.E2.1.Micro", "region": "ap-singapore-1", "vcpu": 1, "ram_gb": 1},
    "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "split": {"seed": SEED, "train": len(X_train), "validation": len(X_val), "test": len(X_test)},
    "data_load_seconds": round(data_load_seconds, 3),
    "training_seconds": round(training_seconds, 3),
    "best_iteration": int(model.best_iteration_),
    "auc_roc": round(float(roc_auc_score(y_test, proba)), 4),
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "f1": round(float(f1_score(y_test, pred)), 4),
    "precision": round(float(precision_score(y_test, pred)), 4),
    "recall": round(float(recall_score(y_test, pred)), 4),
    "latency_1_row_ms": round(latency_1_row_ms, 3),
    "latency_repeats": LATENCY_REPEATS,
    "throughput_1000_rows_seconds_median": round(batch_seconds, 5),
    "throughput_1000_rows_per_second": round(1000 / batch_seconds, 1),
    "throughput_repeats": BATCH_REPEATS,
}

with open("benchmark_result.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))

# =====================================================================
#  2 - TRAINING, VALIDATION AND TESTING
# Models: Linear Regression, Random Forest, XGBoost and LSTM (forecasters),
#         Isolation Forest (anomaly detector)
#
# Reads the model-ready tables made by  01_preprocess_data.py  and, for every group:
#   * cuts the data IN TIME ORDER into three parts: the latest 20% is TEST; the earlier 80% is split
#     70% TRAIN / 30% VALIDATION (= 56% / 24% / 20% of all rows); the last 15 minutes before each cut are removed
#   * Random Forest settings are chosen on the validation part, then the model is refitted and scored ONCE on test
#   * XGBoost: early stopping on the validation part picks the number of trees
#   * LSTM: looks at the last LSTM_WINDOW grid steps of each vehicle (scored only on rows with a full history)
#   * writes the tables: every model's MAE / RMSE / R2, the lowest-MAE model, the best model vs the baseline,
#     the actual-vs-predicted plots (per group / variant, target and algorithm), importance and Isolation Forest results
#
# Run (VS Code terminal, .venv active):  python3 02_train_validate_test.py
# Write your own version for your project and follow your institution's rules on disclosing AI assistance.
# =====================================================================

import os
import re
import sys
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # save PNG files only; no Tk window, no crash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quieter TensorFlow
try:
    from xgboost import XGBRegressor
    XGB_IMPORT_ERROR = None
except Exception as e:                                  # not installed, or (on a Mac) libomp is missing
    XGBRegressor = None
    XGB_IMPORT_ERROR = f"{type(e).__name__}: {str(e)[:150]}"

warnings.filterwarnings("ignore")

# =====================================================================
# SETTINGS  (edit this block only)
# =====================================================================
RUN_LABEL = "month5"                              # must be the SAME 
PREPARED_FOLDER = None                                # None = the folder "prepared" next to this script
GROUPS_TO_TRAIN = None                                # None = every prepared group, or e.g. ["ALL_BUS", "ALL_all"]

# Split in time order: the latest TEST_PCT % of the data is the test part; the earlier part is
# divided TRAIN_SHARE : VAL_SHARE into training and validation (70 : 30 -> 56 % / 24 % / 20 % of all rows).
TEST_PCT = 20
TRAIN_SHARE, VAL_SHARE = 70, 30          # shares of the part that remains after the test part (add up to 100)
SPLIT_MODE = "per_vehicle"   # "per_vehicle": cut-offs from each vehicle's own rows | "pooled": one set of cut-off times for the whole group
REFIT_ON_TRAIN_VAL = True    # after choosing settings on validation, refit on train + validation
MAE_TARGET = 5.0                         # goal for the test MAE, in degrees C
EVAL_SAMPLE = 500000                     # validation / test scoring uses at most this many random rows
MAX_TRAIN_ROWS = 300000                  # training uses at most this many random rows per model
LOVO_MAX_VEHICLES = 8                    # leave-one-vehicle-out check up to this many vehicles
RUN_UNSEEN_VEHICLE_CHECK = True

# Random Forest
RF_TREES = 120                 # all trees are trained on the training rows; more trees = steadier but slower
RF_GRID = [                    # settings compared on the validation part
    {"max_depth": 8, "min_samples_leaf": 5},
    {"max_depth": 14, "min_samples_leaf": 5},
    {"max_depth": 20, "min_samples_leaf": 10},
    {"max_depth": None, "min_samples_leaf": 20},
]

# XGBoost (needs: python3 -m pip install xgboost ; on a Mac also: brew install libomp)
ENABLE_XGBOOST = True
XGB_GRID = [                   # settings compared on the validation part
    {"max_depth": 4, "learning_rate": 0.1},
    {"max_depth": 6, "learning_rate": 0.1},
    {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 5},
]
XGB_MAX_TREES = 600            # upper limit; early stopping on the validation part picks the real number
XGB_EARLY_STOP = 30

# LSTM (needs: python3 -m pip install tensorflow ; it runs on the CPU unless an NVIDIA GPU is set up)
ENABLE_LSTM = True

LSTM_WINDOW = 90
LSTM_UNITS = 128

LSTM_EPOCHS = 100
LSTM_BATCH = 256
LSTM_PATIENCE = 10

LSTM_MAX_TRAIN = 200000
LSTM_MAX_EVAL = 150000

# Isolation Forest
IF_CONTAMINATION = 0.02
IF_N_ESTIMATORS = 200
RANDOM_SEED = 42

# =====================================================================
# Below this line: no need to edit
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPARED_DIR = Path(PREPARED_FOLDER) if PREPARED_FOLDER else Path(BASE_DIR) / "prepared"


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


if TRAIN_SHARE + VAL_SHARE != 100 or not (5 <= TEST_PCT <= 50):
    print("TRAIN_SHARE + VAL_SHARE must add up to 100, and TEST_PCT should be between 5 and 50.")
    print(f"Now: TRAIN_SHARE={TRAIN_SHARE}, VAL_SHARE={VAL_SHARE}, TEST_PCT={TEST_PCT}")
    sys.exit(1)

# what step 1 prepared
banner("STEP 1 RESULT: PREPARED TABLES")
index_path = PREPARED_DIR / f"prepared_index_{RUN_LABEL}.json"
if not index_path.exists():
    print(f"Not found: {index_path}")
    print("   and use the same RUN_LABEL in both files.")
    sys.exit(1)
INDEX = json.load(open(index_path))
DEVICE_COL = INDEX["device_col"]
TARGETS = INDEX["targets"]
GRID_SEC = INDEX["grid_sec"]
HORIZON_MIN = INDEX["horizon_min"]
entries = INDEX["groups"]
if GROUPS_TO_TRAIN:
    want_ = {str(g).strip().upper() for g in GROUPS_TO_TRAIN}
    entries = [e for e in entries if e["label"].upper() in want_]
    if not entries:
        print(f"None of GROUPS_TO_TRAIN={GROUPS_TO_TRAIN} was prepared. Available:",
              [e["label"] for e in INDEX["groups"]])
        sys.exit(1)
print(pd.DataFrame([{k: e[k] for k in ("label", "rows", "vehicles", "variants")} for e in entries]).to_string(index=False))
print(f"Targets: {TARGETS}   grid: {GRID_SEC} s   horizon: {HORIZON_MIN} min")


def score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    hot = y_true >= np.percentile(y_true, 90)
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE_hottest10pct": float(np.sqrt(mean_squared_error(y_true[hot], y_pred[hot])))}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def split_three(data):
    """Time-ordered train / validation / test, cut by SHARE OF DATA (not by calendar days, so gaps in
    the calendar cannot leave a part empty). The last HORIZON_MIN minutes before a boundary are removed
    from the earlier part, because their labels look into the next part."""
    gap = HORIZON_MIN * 60.0
    secs = (data["_t"] - data["_t"].min()).dt.total_seconds()
    rest = 1.0 - TEST_PCT / 100.0
    f1, f2 = rest * TRAIN_SHARE / 100.0, rest
    if SPLIT_MODE == "per_vehicle":
        grp = secs.groupby(data[DEVICE_COL])
        c1 = grp.transform(lambda x: x.quantile(f1))
        c2 = grp.transform(lambda x: x.quantile(f2))
    else:
        c1, c2 = secs.quantile(f1), secs.quantile(f2)
    train = data[secs < c1 - gap]
    val = data[(secs >= c1) & (secs < c2 - gap)]
    test = data[secs >= c2]
    return train, val, test


def make_rf(params):
    return RandomForestRegressor(n_estimators=RF_TREES, random_state=RANDOM_SEED, n_jobs=-1, **params)


def fit_predict(model, cols, tr, te, target):
    if len(tr) > MAX_TRAIN_ROWS:
        tr = tr.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
    m = clone(model)
    t0 = time.time()
    m.fit(tr[cols], tr[f"{target}_future"])
    return m, m.predict(te[cols]), time.time() - t0


def make_xgb(params, n_estimators, early_stopping=False):
    kw = dict(n_estimators=n_estimators, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              random_state=RANDOM_SEED, n_jobs=-1, **params)
    if early_stopping:
        kw["early_stopping_rounds"] = XGB_EARLY_STOP
    return XGBRegressor(**kw)


def window_ends(part, window):
    """Positions of rows that have a full, gap-free history of `window` grid steps in the same vehicle."""
    secs = (part["_t"] - part["_t"].min()).dt.total_seconds().values
    dev = part[DEVICE_COL].values
    n = len(part)
    ok = np.zeros(n, dtype=bool)
    ok[1:] = (dev[1:] == dev[:-1]) & (np.abs(secs[1:] - secs[:-1] - GRID_SEC) < 0.5)
    idx = np.arange(n)
    last_reset = np.maximum.accumulate(np.where(ok, 0, idx))
    return np.where(idx - last_reset >= window - 1)[0]


def run_lstm(label, train, val, test, inputs):
    """One LSTM predicting the 15-minute change of both temperatures from the last LSTM_WINDOW steps.
    Early stopping on the validation windows; scored once on the test windows."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as e:
        print(f"  LSTM skipped: TensorFlow could not be imported ({type(e).__name__}). Install it with "
              "python3 -m pip install tensorflow (Python 3.9 to 3.13).")
        return []
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    scaler = StandardScaler().fit(train[inputs])
    offs = np.arange(-LSTM_WINDOW + 1, 1)[None, :]

    def prep(part, cap):
        ends = window_ends(part, LSTM_WINDOW)
        if len(ends) > cap:
            ends = np.sort(np.random.RandomState(RANDOM_SEED).choice(ends, cap, replace=False))
        if len(ends) < 100:
            return None
        xs = scaler.transform(part[inputs]).astype("float32")
        X = xs[ends[:, None] + offs]
        now = part[TARGETS].values[ends]
        fut = part[[f"{t}_future" for t in TARGETS]].values[ends]
        return X, now, fut

    tr, va, te = prep(train, LSTM_MAX_TRAIN), prep(val, LSTM_MAX_EVAL), prep(test, LSTM_MAX_EVAL)
    if tr is None or va is None or te is None:
        print(f"  LSTM skipped: fewer than 100 rows with a full {LSTM_WINDOW}-step history in one of the parts.")
        return []
    (Xtr, now_tr, fut_tr), (Xva, now_va, fut_va), (Xte, now_te, fut_te) = tr, va, te
    d_tr = fut_tr - now_tr
    mu, sd = d_tr.mean(axis=0), d_tr.std(axis=0) + 1e-6          # the network predicts the scaled change
    model = keras.Sequential([
        keras.layers.Input(shape=(LSTM_WINDOW, len(inputs))),
        keras.layers.LSTM(LSTM_UNITS),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(len(TARGETS)),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss=keras.losses.Huber())#new: smaller steps (default is 1e-3)# new: spikes disturb it less than "mse"
    callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE,
                                  min_delta=1e-4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=3, min_lr=1e-5),       # new: halves the step when stuck
]
    t0 = time.time()
    hist = model.fit(Xtr, (d_tr - mu) / sd, validation_data=(Xva, (fut_va - now_va - mu) / sd),
                     epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, callbacks=callbacks, verbose=0)
    secs = time.time() - t0
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    print(f"epochs run: {len(hist.history['loss'])}, best epoch: {best_epoch}")

    def predict(X, now):
        return now + model.predict(X, batch_size=2048, verbose=0) * sd + mu

    pv, pt = predict(Xva, now_va), predict(Xte, now_te)
    settings = (f"window={LSTM_WINDOW} steps ({LSTM_WINDOW * GRID_SEC // 60} min), units={LSTM_UNITS}, "
                f"epochs used={len(hist.history['loss'])}, trained on {len(Xtr)} windows")
    out = []
    for j, t in enumerate(TARGETS):
        out.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "Persistence baseline (LSTM rows)",
                    "RMSE_val": round(rmse(fut_va[:, j], now_va[:, j]), 3), **score(fut_te[:, j], now_te[:, j]),
                    "Settings": "", "Train_sec": 0.0, "Test_rows": len(Xte)})
        out.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "LSTM",
                    "RMSE_val": round(rmse(fut_va[:, j], pv[:, j]), 3), **score(fut_te[:, j], pt[:, j]),
                    "Settings": settings, "Train_sec": round(secs, 1), "Test_rows": len(Xte)})
    print(f"  LSTM: {settings}; scored on {len(Xte)} test windows")
    return out


# =====================================================================
# TRAIN, VALIDATE AND TEST ONE GROUP
# =====================================================================
def train_group(entry):
    label, safe, inputs = entry["label"], entry["safe"], entry["inputs"]
    data = pd.read_pickle(entry["file"])
    data[DEVICE_COL] = data[DEVICE_COL].astype(object)
    banner(f"GROUP: {label}   ({data[DEVICE_COL].nunique()} vehicles, {len(data)} prepared rows)")
    print(f"  inputs ({len(inputs)}): {inputs}")

    # ---- split in time order: train / validation / test ----
    train, val, test = split_three(data)
    val_full, test_full = val, test            # the LSTM needs the parts without sampling
    print(f"  split in time order (test = latest {TEST_PCT}%; the rest is {TRAIN_SHARE}% train / {VAL_SHARE}% validation): "
          f"train {len(train)} | validation {len(val)} | test {len(test)} rows")
    if min(len(train), len(val), len(test)) < 50:
        print("  Skipping: one of the three parts has fewer than 50 rows. Use more data "
              "or a different split.")
        return None
    # keep what is needed later, then free the big table
    all_vehicles = data[DEVICE_COL].unique()
    data_s = data if len(data) <= 3 * MAX_TRAIN_ROWS else data.sample(3 * MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
    del data
    if len(val) > EVAL_SAMPLE:
        val = val.sample(EVAL_SAMPLE, random_state=RANDOM_SEED).sort_index()
        print(f"  validation scored on a random sample of {EVAL_SAMPLE} rows")
    if len(test) > EVAL_SAMPLE:
        test = test.sample(EVAL_SAMPLE, random_state=RANDOM_SEED).sort_index()
        print(f"  test scored on a random sample of {EVAL_SAMPLE} rows")
    tr_cap = train if len(train) <= MAX_TRAIN_ROWS else train.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
    trval = pd.concat([tr_cap, val]) if REFIT_ON_TRAIN_VAL else tr_cap      # the models sample from this again

    rows, grid_rows, xgb_grid_rows, preds, importances = [], [], [], {}, {}
    models_final = {}
    for t in TARGETS:
        y_val, y_test = val[f"{t}_future"].values, test[f"{t}_future"].values
        base = score(y_test, test[t].values)
        rows.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "Persistence baseline",
                     "RMSE_val": round(rmse(y_val, val[t].values), 3), **base,
                     "Settings": "", "Train_sec": 0.0})

        # Linear Regression
        lr = make_pipeline(StandardScaler(), LinearRegression())
        _, pv, _ = fit_predict(lr, inputs, train, val, t)
        m, pt, secs = fit_predict(lr, inputs, trval, test, t)
        rows.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "Linear Regression",
                     "RMSE_val": round(rmse(y_val, pv), 3), **score(y_test, pt),
                     "Settings": "", "Train_sec": round(secs, 1)})
        preds[(t, "Linear Regression")] = pt
        models_final[(t, "Linear Regression")] = lr
        importances[(t, "LR")] = pd.Series(m[-1].coef_, index=inputs)

        # Random Forest: choose settings on validation, then score once on test
        best = None
        for params in RF_GRID:
            _, pv, _ = fit_predict(make_rf(params), inputs, train, val, t)
            r = rmse(y_val, pv)
            grid_rows.append({"Group": label, "Target": t, "Settings": str(params), "RMSE_val": round(r, 3)})
            if best is None or r < best[0]:
                best = (r, params)
        m, pt, secs = fit_predict(make_rf(best[1]), inputs, trval, test, t)
        rows.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "Random Forest",
                     "RMSE_val": round(best[0], 3), **score(y_test, pt),
                     "Settings": str(best[1]), "Train_sec": round(secs, 1)})
        preds[(t, "Random Forest")] = pt
        models_final[(t, "Random Forest")] = make_rf(best[1])
        importances[(t, "RF")] = pd.Series(m.feature_importances_, index=inputs)
        print(f"  {t}: best Random Forest settings on validation = {best[1]} (validation RMSE {best[0]:.3f})")

        # XGBoost: early stopping on the validation part, then refit and score once on test
        if ENABLE_XGBOOST and XGBRegressor is not None:
            tr_x = train if len(train) <= MAX_TRAIN_ROWS else train.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
            best_x = None
            for params in XGB_GRID:
                mx = make_xgb(params, XGB_MAX_TREES, early_stopping=True)
                mx.fit(tr_x[inputs], tr_x[f"{t}_future"], eval_set=[(val[inputs], y_val)], verbose=False)
                n_best = int(getattr(mx, "best_iteration", XGB_MAX_TREES - 1)) + 1
                r = rmse(y_val, mx.predict(val[inputs], iteration_range=(0, n_best)))
                xgb_grid_rows.append({"Group": label, "Target": t, "Settings": str(params),
                                      "Trees_chosen": n_best, "RMSE_val": round(r, 3)})
                if best_x is None or r < best_x[0]:
                    best_x = (r, params, n_best)
            m, pt, secs = fit_predict(make_xgb(best_x[1], best_x[2]), inputs, trval, test, t)
            rows.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "XGBoost",
                         "RMSE_val": round(best_x[0], 3), **score(y_test, pt),
                         "Settings": str({**best_x[1], "n_estimators": best_x[2]}), "Train_sec": round(secs, 1)})
            preds[(t, "XGBoost")] = pt
            models_final[(t, "XGBoost")] = make_xgb(best_x[1], best_x[2])
            importances[(t, "XGB")] = pd.Series(m.feature_importances_, index=inputs)
            print(f"  {t}: best XGBoost settings on validation = {best_x[1]} with {best_x[2]} trees "
                  f"(validation RMSE {best_x[0]:.3f})")
    if ENABLE_LSTM:
        rows.extend(run_lstm(label, train, val_full, test_full, inputs))
    res = pd.DataFrame(rows)
    res["Test_rows"] = res["Test_rows"].fillna(len(test)).astype(int) if "Test_rows" in res.columns else len(test)
    pd.DataFrame(grid_rows).to_csv(os.path.join(BASE_DIR, f"out_rf_validation_{safe}_{RUN_LABEL}.csv"), index=False)
    if xgb_grid_rows:
        pd.DataFrame(xgb_grid_rows).to_csv(os.path.join(BASE_DIR, f"out_xgb_validation_{safe}_{RUN_LABEL}.csv"), index=False)

    # ---- which inputs matter ----
    imp_rows = []
    for (t, kind), s_ in importances.items():
        top = s_.reindex(s_.abs().sort_values(ascending=False).index).head(8)
        print(f"\n  Top inputs, {kind} ({'scaled coefficients' if kind == 'LR' else 'importance'}), {t}:")
        print(top.round(3).to_string())
        for name, v in s_.items():
            imp_rows.append({"Group": label, "Target": t, "Model": kind, "Input": name, "Value": round(float(v), 4)})
    pd.DataFrame(imp_rows).to_csv(os.path.join(BASE_DIR, f"out_importance_{safe}_{RUN_LABEL}.csv"), index=False)

    # ---- unseen-vehicle check ----
    vehicles = all_vehicles
    folds = []
    if RUN_UNSEEN_VEHICLE_CHECK and len(vehicles) >= 2:
        if len(vehicles) <= LOVO_MAX_VEHICLES:
            folds = [[v] for v in vehicles]
        elif len(vehicles) >= 5:
            rng = np.random.RandomState(RANDOM_SEED)
            folds = [list(rng.choice(vehicles, size=max(1, int(len(vehicles) * 0.2)), replace=False))]
    extra = []
    for t in TARGETS:
        for name in ["Linear Regression", "Random Forest"] + (["XGBoost"] if (t, "XGBoost") in models_final else []):
            r = []
            for hold in folds:
                h = data_s[DEVICE_COL].isin(hold)
                if h.sum() < 50 or (~h).sum() < 50:
                    continue
                _, p, _ = fit_predict(models_final[(t, name)], inputs, data_s[~h], data_s[h], t)
                r.append(rmse(data_s.loc[h, f"{t}_future"], p))
            if r:
                extra.append({"Target": t, "Model": name, "RMSE_unseen_vehicles": round(float(np.mean(r)), 3),
                              "RMSE_worst_unseen_vehicle": round(float(np.max(r)), 3)})
    if extra:
        res = res.merge(pd.DataFrame(extra), on=["Target", "Model"], how="left")
    res = res.round(3)
    res["MAE_below_target"] = res["MAE"] < MAE_TARGET

    # ---- Isolation Forest ----
    if_cols = [c for c in inputs if c != "is_derating"] + [t for t in TARGETS if t not in inputs]
    train_if = train if len(train) <= MAX_TRAIN_ROWS else train.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
    iso = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION,
                          random_state=RANDOM_SEED, n_jobs=-1).fit(train_if[if_cols])
    a_score = -iso.score_samples(test[if_cols])
    flag = iso.predict(test[if_cols]) == -1
    flag_val = iso.predict(val[if_cols]) == -1
    auc = np.nan
    if "is_derating" in test.columns and test["is_derating"].nunique() == 2:
        auc = roc_auc_score(test["is_derating"].values, a_score)
    if_rows = []
    for t in TARGETS:
        change = test[f"{t}_future"].values - test[t].values
        hot = test[f"{t}_future"].values >= np.percentile(test[f"{t}_future"].values, 90)
        if_rows.append({
            "Run": RUN_LABEL, "Group": label, "Target": t, "Test rows": len(test),
            "Flagged % (validation)": round(100 * flag_val.mean(), 2),
            "Flagged % (test)": round(100 * flag.mean(), 2),
            "Mean 15-min change if flagged": round(float(change[flag].mean()), 2) if flag.any() else np.nan,
            "Mean 15-min change if normal": round(float(change[~flag].mean()), 2),
            "Hot in 15 min % if flagged": round(100 * float(hot[flag].mean()), 1) if flag.any() else np.nan,
            "Hot in 15 min % if normal": round(100 * float(hot[~flag].mean()), 1),
            "AUC vs derating": round(float(auc), 3) if auc == auc else np.nan})
    if_res = pd.DataFrame(if_rows)
    found_rows = test.loc[flag, [DEVICE_COL, "_t"] + TARGETS].copy()
    found_rows["anomaly_score"] = a_score[flag]
    found_rows.sort_values("anomaly_score", ascending=False).to_csv(
        os.path.join(BASE_DIR, f"out_anomalies_{safe}_{RUN_LABEL}.csv"), index=False)

    # ---- plot: actual vs predicted, every target x every algorithm (group / variant in the title) ----
    model_names = [m for m in ["Linear Regression", "Random Forest", "XGBoost"]
                   if (TARGETS[0], m) in preds]
    n = min(1000, len(test))
    idx = np.linspace(0, len(test) - 1, n).astype(int)      # spread over the whole test part
    fig, axes = plt.subplots(len(TARGETS), len(model_names),
                             figsize=(6 * len(model_names), 3.8 * len(TARGETS)), squeeze=False)
    for i, t in enumerate(TARGETS):
        for j, mn in enumerate(model_names):
            ax = axes[i][j]
            r = res[(res["Target"] == t) & (res["Model"] == mn)].iloc[0]
            ax.plot(test[f"{t}_future"].values[idx], label=f"Actual (t+{HORIZON_MIN} min)", lw=1)
            ax.plot(preds[(t, mn)][idx], label="Predicted", lw=1, alpha=0.8)
            ax.set_title(f"{label} | {t} | {mn}\nMAE {r['MAE']:.2f}   RMSE {r['RMSE']:.2f}", fontsize=9)
            ax.set_xlabel("Test sample")
            ax.set_ylabel("Temperature (°C)")
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"out_plot_{safe}_{RUN_LABEL}.png"), dpi=120)
    plt.close(fig)

    return res, if_res




# =====================================================================
# RUN: every prepared group, then the result tables
# =====================================================================
if ENABLE_XGBOOST and XGBRegressor is None:
    print(f"XGBoost skipped: {XGB_IMPORT_ERROR}")
    #print("Install with  python3 -m pip install xgboost  (on a Mac also: brew install libomp).")

all_res, all_if = [], []
for entry in entries:
    out = train_group(entry)
    if out is not None:
        all_res.append(out[0])
        all_if.append(out[1])

if not all_res:
    print("\nNo group produced results. Read the messages above (usually too little continuous data).")
    sys.exit(1)

results = pd.concat(all_res, ignore_index=True)
results.to_csv(os.path.join(BASE_DIR, f"out_results_{RUN_LABEL}.csv"), index=False)
if_all = pd.concat(all_if, ignore_index=True)
if_all.to_csv(os.path.join(BASE_DIR, f"out_isolation_forest_{RUN_LABEL}.csv"), index=False)

banner("FORECAST RESULTS (test part; lower RMSE / MAE is better, higher R2 is better)")
print("RMSE_val = error on the validation part (used to choose settings).")
print("RMSE, MAE, R2, RMSE_hottest10pct = error on the test part, scored once.")
for (g_, t), sub in results.groupby(["Group", "Target"], sort=False):
    print(f"\n--- {g_} | {t} ---")
    print(sub.sort_values("RMSE").drop(columns=["Run", "Group", "Target"]).to_string(index=False))

banner(f"SUMMARY: every model per group and target (goal: test MAE below {MAE_TARGET})")
summ = []
for (g_, t), sub in results.groupby(["Group", "Target"], sort=False):
    base_all = sub[sub["Model"] == "Persistence baseline"].iloc[0]
    base_l = sub[sub["Model"] == "Persistence baseline (LSTM rows)"]
    for _, r in sub.iterrows():
        if str(r["Model"]).startswith("Persistence"):
            continue
        base = base_l.iloc[0] if (r["Model"] == "LSTM" and len(base_l)) else base_all
        summ.append({"Group": g_, "Target": t, "Model": r["Model"], "Test MAE": r["MAE"], "Test RMSE": r["RMSE"],
                     "Baseline MAE": base["MAE"], "Baseline RMSE": base["RMSE"],
                     "RMSE vs baseline": round(r["RMSE"] / base["RMSE"], 3) if base["RMSE"] else np.nan,
                     "MAE below target": bool(r["MAE"] < MAE_TARGET),
                     "Beats baseline": bool(r["RMSE"] < base["RMSE"]), "Test rows": r["Test_rows"]})
summ = pd.DataFrame(summ).sort_values(["Group", "Target", "RMSE vs baseline"]).reset_index(drop=True)
print(summ.to_string(index=False))
summ.to_csv(os.path.join(BASE_DIR, f"out_summary_best_{RUN_LABEL}.csv"), index=False)
print("Within each group and target the first row is the best model (lowest RMSE compared with its own baseline).")
print("A result counts only when BOTH 'MAE below target' and 'Beats baseline' are True.")
print("The LSTM is scored only on rows with a full history, so it is compared with a baseline on those same rows.")

banner(f"MAE OF EVERY MODEL (test part, degrees C; lower is better; goal below {MAE_TARGET})")
mae_tab = summ.pivot_table(index=["Group", "Target"], columns="Model", values="Test MAE", sort=False)
model_cols = list(mae_tab.columns)
base_mae = results[results["Model"] == "Persistence baseline"].set_index(["Group", "Target"])["MAE"]
mae_tab.insert(0, "Persistence baseline", base_mae.reindex(mae_tab.index))
mae_tab["Lowest MAE model"] = mae_tab[model_cols].idxmin(axis=1)
mae_tab["Lowest MAE"] = mae_tab[model_cols].min(axis=1)
mae_tab["Below target"] = mae_tab["Lowest MAE"] < MAE_TARGET
print(mae_tab.round(3).to_string())
mae_tab.round(3).to_csv(os.path.join(BASE_DIR, f"out_mae_table_{RUN_LABEL}.csv"))
print("The LSTM MAE is measured on rows with a full history, so compare it with its own baseline in the SUMMARY table.")

banner("BEST ALGORITHM PER GROUP AND TARGET")
best_tab = summ.groupby(["Group", "Target"], sort=False).head(1)
print(best_tab[["Group", "Target", "Model", "Test MAE", "Test RMSE", "Baseline MAE", "Baseline RMSE",
                "RMSE vs baseline", "MAE below target", "Beats baseline"]].to_string(index=False))
best_tab.to_csv(os.path.join(BASE_DIR, f"out_best_algorithm_{RUN_LABEL}.csv"), index=False)

banner("ISOLATION FOREST (unsupervised: compare flagged vs normal rows, no RMSE)")
print(if_all.to_string(index=False))
print("\nA forecaster counts only if its RMSE is below the Persistence baseline.")
print("If 'Flagged % (test)' is far above the setting IF_CONTAMINATION, the test period looks")
print("different from the training period; that is not a fault rate.")
banner("DONE - files starting with out_ are in the folder of this script")
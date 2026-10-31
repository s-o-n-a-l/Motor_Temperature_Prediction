# =====================================================================
# 12M BUS (4-wheeler), SEVERAL MONTHLY CSV FILES: motor + MCU temperature, 15 minutes ahead
# Models: Linear Regression, Random Forest, XGBoost, LSTM, Isolation Forest
#
#   * All CSV files listed in CSV_FILES are read (in chunks) and joined; only the
#     selected variants' vehicles and the needed columns are kept, to save memory.
#   * The data is cut IN TIME ORDER into three parts: the latest 20% is TEST; the
#     earlier 80% is split 70% TRAIN / 30% VALIDATION (= 56% / 24% / 20% of all rows).
#   * Random Forest / XGBoost settings are chosen on the validation part, then the model is
#     refitted and scored ONCE on the test part. Every tree is trained on training rows.
#   * The LSTM looks at a window of the last LSTM_SEQ_LEN grid steps and predicts the 15-min
#     temperature CHANGE; its number of epochs is chosen on the validation part.
#   * The result table says whether the test MAE is below MAE_TARGET and whether the
#     model beats the "temperature stays the same" baseline.
#
# Needs:  pip install xgboost torch
#
# Use this to get first numbers and to learn the pipeline. Write your own
# version of the code for your project, and follow your institution's rules
# on disclosing AI assistance.
#
# Run (VS Code terminal, .venv active):  python3 motor_mcu_temp_3w.py
# =====================================================================

import os
import re
import sys
import json
import ast
import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score)

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# 1. SETTINGS  (edit this block only)
# =====================================================================
RUN_LABEL = "months1to5"                              # change for each new batch of months
CSV_FILES = [   Path("/Users/sonal/Downloads/3b317d8e-7ff2-4b74-8ee4-87d05472f0ae.csv"),
            Path("/Users/sonal/Downloads/99ef19a2-ede4-4b96-a6cd-4e543a817b82.csv"),
            Path("/Users/sonal/Downloads/cc716fa0-4b89-4c44-9c34-4f4f419f016d.csv"),
            Path("/Users/sonal/Downloads/3f7c1a42-af34-4042-b35c-05865d1acf1d.csv"),
            Path("/Users/sonal/Downloads/16bef7ec-480b-40e4-8519-53a74751d8dc.csv")                                      # the monthly CSV files; all are joined

]
CSV_FOLDER = None                                     # or a folder: every .csv inside is used (when CSV_FILES is empty)
CHUNK_ROWS = 500000                                   # files are read this many rows at a time
VEHICLE_LIST_CSV = Path("/Users/sonal/Downloads/CAN Signals Info - updated.xlsx")   # vehicle list (.xlsx or .csv)
VEHICLE_LIST_SHEET = 0                                # sheet name or number (only used for .xlsx)
VEHICLE_LIST_DEVICE_COL = "device_id"
VEHICLE_LIST_VARIANT_COL = "variant"

VARIANTS = ["12M","9M","3s","6s","55T"]                                    # variants to model (text must match the vehicle list)
CLASS_LABEL = "4W"                                    # name used in the results (groups: 4W_all, 4W_12M, ...)
RUN_CLASS_LEVEL = True                                # one model group for all variants together
RUN_VARIANT_LEVEL = True                              # plus one group per variant (skipped when there is only one variant)

# Column names in the downloaded files (matched ignoring case / underscores / spaces)
DEVICE_COL = "device_id"
TIMESTAMP_COL = "timestamp"                           # used if it parses; otherwise date + time below
DATE_COL = "date"
CLOCK_COL = "time"

# What to predict. For each target the FIRST name that exists with data is used.
TARGET_CANDIDATES = [
    ("Motor", ["Motor_Temperature", "MCU_Motor_Temp"]),
    ("MCU", ["MCU_Temperature"]),
]

# Number inputs (spelling as in your data). Names not found are skipped with a message.
# Left out on purpose: Vehicle_DTE_Km and MCU_Odometer_Val (they identify a vehicle or
# just grow with time, so a model can learn the vehicle instead of the physics).
NUMERIC_FEATURES = [
    "MCU_HighPowerVoltage", "MCU_HighPowerCurrent", "MCU_Motor_Speed",
    "Torque_Command", "MCU_Motor_Torque_Estimated", "Traction_Torque_Avaliable",
    "MCU_Throttle_perc", "MCU_Brake_perc", "MCU_Motor_RPM", "MCU_Speed_Kmph",
    "Ambient_Temprature",                                                    # skipped / dropped if absent or mostly empty
    "Pump_Speed", "Pump_Voltage", "Pump_Current", "TCS_Pump_Temperature",   # cooling signals, same rule
]
DERATING = ["Traction_Derating", "Regen_Derating"]    # combined into one 0/1 flag
CATEGORICAL = ["MCU_Drive_Mode", "HandBreakStatus", "ignition"]   # state signals -> 0/1 columns
PACKET_FIELDS = ["packet_type", "packet_status"]      # printed for inspection; see next line
USE_PACKET_FIELDS_AS_INPUTS = False                   # False = use them for cleaning only (recommended)

# Rows to drop or keep by state. Values are compared as upper-case text.
# Look at the printed value counts first, then decide. Examples:
#   "ignition": {"include": ["1"]},   "MCU_Drive_Mode": {"exclude": ["0"]},
#   "packet_status": {"include": ["OK"]}
STATE_FILTERS = {}

# Split in time order: the latest TEST_PCT % of the data is the test part; the earlier part is
# divided TRAIN_SHARE : VAL_SHARE into training and validation (70 : 30 -> 56 % / 24 % / 20 % of all rows).
TEST_PCT = 20
TRAIN_SHARE, VAL_SHARE = 70, 30          # shares of the part that remains after the test part (add up to 100)
MAE_TARGET = 5.0                         # goal for the test MAE, in degrees C
EVAL_SAMPLE = 500000                     # validation / test scoring uses at most this many random rows
MAX_DAYS_FROM_MEDIAN = 200               # packets dated farther than this from the middle of the data are dropped
VALID_FROM, VALID_TO = None, None        # optional hard limits, e.g. "2026-01-01", "2026-06-30"
PREFER_DATE_TIME = True                  # time = packet generation time (date + time columns) before 'timestamp'
SPLIT_MODE = "per_vehicle"   # "per_vehicle": cut-offs from each vehicle's own rows | "pooled": one set of cut-off times for the whole group
REFIT_ON_TRAIN_VAL = True    # after choosing settings on validation, refit on train + validation

# Random Forest
RF_TREES = 120                 # all trees are trained on the training rows; more trees = steadier but slower
RF_GRID = [                  # settings compared on the validation part
    {"max_depth": 8, "min_samples_leaf": 5},
    {"max_depth": 14, "min_samples_leaf": 5},
    {"max_depth": 20, "min_samples_leaf": 10},
    {"max_depth": None, "min_samples_leaf": 20},
]

# XGBoost
RUN_XGBOOST = True
XGB_TREES = 400
XGB_GRID = [                 # settings compared on the validation part
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 10},
    {"max_depth": 8, "learning_rate": 0.03, "min_child_weight": 20},
]

# LSTM (sees a window of the last LSTM_SEQ_LEN grid steps, not a single row)
RUN_LSTM = True
LSTM_SEQ_LEN = 30            # 30 x 10 s = 5 minutes of history
LSTM_STRIDE = 6              # one window every 6 grid steps (1 min); smaller = more windows, slower
LSTM_MAX_SEQ = 200000        # max windows used for training / validation / test
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.1           # only used when LSTM_LAYERS > 1
LSTM_EPOCHS = 15
LSTM_PATIENCE = 3            # stop if validation RMSE does not improve for this many epochs
LSTM_BATCH = 512
LSTM_LR = 1e-3

# Time handling
GRID_SEC = 10            # packets are put on a regular 10-second grid
FILL_SEC = 60            # a signal is carried forward at most this long
HORIZON_MIN = 15
MIN_COVERAGE = 0.5       # a signal present in less than 50% of a group's rows is dropped
MIN_ROWS_PER_GROUP = 300
MAX_TRAIN_ROWS = 300000
LOVO_MAX_VEHICLES = 8    # leave-one-vehicle-out check up to this many vehicles
RUN_UNSEEN_VEHICLE_CHECK = True

# Exploration (summary statistics, distributions, correlations)
RUN_EDA = True
EDA_MAX_SAMPLE = 200000
HIGH_CORR = 0.95

# Isolation Forest
IF_CONTAMINATION = 0.02
IF_N_ESTIMATORS = 200
RANDOM_SEED = 42

# =====================================================================
# Below this line: no need to edit
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STEPS_AHEAD = int(HORIZON_MIN * 60 / GRID_SEC)
FILL_ROWS = max(1, int(FILL_SEC / GRID_SEC))
MISSING_TEXT = {"NAN", "NONE", "", "N", "NA", "<NA>", "NULL"}
TARGETS = []             # filled in after the data is loaded


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


def clean_id(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def to_datetime_flexible(values):
    s_ = pd.Series(values)
    num = pd.to_numeric(s_, errors="coerce")
    if num.notna().mean() > 0.9:
        med = num.dropna().abs().median()
        unit = "us" if med > 1e14 else ("ms" if med > 1e11 else "s")
        return pd.to_datetime(num, unit=unit, errors="coerce")
    return pd.to_datetime(s_, errors="coerce")


def parse_utc_datetime(date_s, time_s):
    """Combine separate date and time columns. Tries common formats (05012026 / 081530,
    2026-01-05, 05/01/2026, 12-hour clock ...), strips 'UTC'/'Z', restores lost leading zeros."""
    d = date_s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    t = time_s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    d = d.str.replace(r"\s*(UTC|Z)$", "", regex=True)
    t = t.str.replace(r"\s*(UTC|Z)$", "", regex=True)
    if d.str.fullmatch(r"\d+").mean() > 0.9:
        d = d.str.zfill(8 if d.str.len().max() >= 7 else 6)
    if t.str.fullmatch(r"\d+").mean() > 0.9:
        t = t.str.zfill(6)
    date_fmts = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d%m%y",
                 "%d%m%Y", "%y%m%d", "%Y%m%d", "%d-%m-%y", "%d/%m/%y", "%m/%d/%y"]
    time_fmts = ["%H:%M:%S", "%H%M%S", "%I:%M:%S %p", "%H:%M:%S.%f", "%H:%M"]
    both = (d + " " + t).dropna()
    sample = both.sample(min(2000, len(both)), random_state=0) if len(both) else both
    best, best_rate = None, 0.0
    for df_ in date_fmts:
        for tf_ in time_fmts:
            fmt = f"{df_} {tf_}"
            rate = pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean() if len(sample) else 0
            if rate > best_rate:
                best, best_rate = fmt, rate
    if best is None or best_rate < 0.9:
        print("  Could not match the date/time format. Sample raw values:")
        print(pd.DataFrame({"date": date_s.head(5).values, "time": time_s.head(5).values}).to_string(index=False))
        return pd.Series(pd.NaT, index=date_s.index)
    print(f"  date/time format detected: '{best}'  ({100 * best_rate:.0f}% of sampled rows parse)")
    return pd.to_datetime(d + " " + t, format=best, errors="coerce")


def state_text(series):
    """Text version of a state signal; missing markers become real NaN."""
    def one(v):
        if v is None or (isinstance(v, float) and v != v):
            return np.nan
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        t = str(v).strip().upper()
        return np.nan if t in MISSING_TEXT else t
    return series.map(one)


def parse_cell(x):
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and x != x):
        return {}
    try:
        return json.loads(x)
    except (TypeError, ValueError):
        pass
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return {}


# =====================================================================
# 2. VEHICLE LIST -> THE SELECTED VEHICLES
# =====================================================================
banner("STEP 1: VEHICLE LIST")
if str(VEHICLE_LIST_CSV).lower().endswith((".xlsx", ".xls")):
    vl = pd.read_excel(VEHICLE_LIST_CSV, sheet_name=VEHICLE_LIST_SHEET)   # needs: pip install openpyxl
else:
    vl = pd.read_csv(VEHICLE_LIST_CSV)
vcols = {norm(c): c for c in vl.columns}
vdev, vvar = vcols.get(norm(VEHICLE_LIST_DEVICE_COL)), vcols.get(norm(VEHICLE_LIST_VARIANT_COL))
if not vdev or not vvar:
    print("Vehicle list columns found:", vl.columns.tolist())
    print("Set VEHICLE_LIST_DEVICE_COL and VEHICLE_LIST_VARIANT_COL.")
    sys.exit(1)
vl["_id"] = clean_id(vl[vdev])
vl["_var"] = vl[vvar].astype(str).str.strip()
print("Variants in the vehicle list (vehicles each):")
print(vl["_var"].value_counts().to_string())
want = {norm(v) for v in VARIANTS}
vl_sel = vl[vl["_var"].map(norm).isin(want)]
if vl_sel.empty:
    print(f"\nNone of {VARIANTS} appear in the list above. Fix VARIANTS so the text matches.")
    sys.exit(1)
vmap = dict(zip(vl_sel["_id"], vl_sel["_var"].str.upper()))
keep_ids = set(vmap)
print(f"\nVehicles selected: {len(keep_ids)}")

# =====================================================================
# 3. LOAD THE CSV FILES (in chunks; keep only the selected vehicles and the needed columns)
# =====================================================================
banner("STEP 2: LOAD CSV FILES")
files = [Path(f) for f in CSV_FILES] if CSV_FILES else sorted(Path(CSV_FOLDER).rglob("*.csv")) if CSV_FOLDER else []
if not files:
    print("No CSV files given. Fill CSV_FILES (or CSV_FOLDER) in the SETTINGS block.")
    sys.exit(1)
not_found = [str(f) for f in files if not f.exists()]
if not_found:
    print("These files do not exist (check the paths, one pair of quotes only):")
    print("\n".join(not_found))
    sys.exit(1)

all_target_names = [n for _, cands in TARGET_CANDIDATES for n in cands]
STATE_SIGNALS = list(dict.fromkeys(CATEGORICAL + PACKET_FIELDS + list(STATE_FILTERS.keys())))
wanted_all = list(dict.fromkeys(all_target_names + NUMERIC_FEATURES + DERATING + STATE_SIGNALS))

frames, total_rows, kept_rows = [], 0, 0
for f in files:
    try:
        header = pd.read_csv(f, nrows=0).columns
    except UnicodeDecodeError:
        print(f"{f.name} is not plain UTF-8 text. Run:  file \"{f}\"  to see what it is.")
        sys.exit(1)
    cm = {norm(c): c for c in header}
    if norm(DEVICE_COL) not in cm:
        print(f"{f.name}: no '{DEVICE_COL}' column. Columns: {list(header)[:30]}")
        sys.exit(1)
    use = [cm[norm(n)] for n in [DEVICE_COL, TIMESTAMP_COL, DATE_COL, CLOCK_COL] + wanted_all if norm(n) in cm]
    if any(norm(s) not in cm for s in wanted_all) and norm("can_data") in cm:
        use.append(cm[norm("can_data")])
    use = list(dict.fromkeys(use))
    n_file = k_file = 0
    for chunk in pd.read_csv(f, usecols=use, chunksize=CHUNK_ROWS, low_memory=False):
        n_file += len(chunk)
        chunk = chunk[clean_id(chunk[cm[norm(DEVICE_COL)]]).isin(keep_ids)]
        k_file += len(chunk)
        if len(chunk):
            frames.append(chunk)
    total_rows += n_file
    kept_rows += k_file
    print(f"  {f.name}: {n_file} rows read, {k_file} belong to the selected vehicles")
if not frames:
    print("No telemetry row belongs to the selected vehicles. Check that device_id looks the same in")
    print("the telemetry files and in the vehicle list (same digits, no spaces).")
    sys.exit(1)
raw = pd.concat(frames, ignore_index=True)
del frames
print(f"Together: {kept_rows} of {total_rows} rows kept, {len(raw.columns)} columns")

colmap = {norm(c): c for c in raw.columns}
sig = pd.DataFrame(index=raw.index)
sig[DEVICE_COL] = clean_id(raw[colmap[norm(DEVICE_COL)]])
found, missing = [], []
for s_ in wanted_all:
    if norm(s_) in colmap:
        sig[s_] = raw[colmap[norm(s_)]]
        found.append(s_)
    else:
        missing.append(s_)

# signals still missing: try to read them from a can_data text column, if there is one
if missing and norm("can_data") in colmap:
    print("Reading missing signals from the can_data column ...")
    parsed = raw[colmap[norm("can_data")]].map(parse_cell)
    keymap = {}
    for d in parsed.head(2000):
        for k in d.keys():
            keymap.setdefault(norm(k), k)
    for s_ in list(missing):
        if norm(s_) in keymap:
            k = keymap[norm(s_)]
            sig[s_] = parsed.map(lambda d: d.get(k))
            found.append(s_)
            missing.remove(s_)
for s_ in missing:
    print(f"  NOT FOUND: {s_}")

# time: packet generation time from the date and time columns (before 'timestamp')
t_col = colmap.get(norm(TIMESTAMP_COL))
d_col, c_col = colmap.get(norm(DATE_COL)), colmap.get(norm(CLOCK_COL))
tt, source = None, None
for kind in (("dt", "ts") if PREFER_DATE_TIME else ("ts", "dt")):
    if kind == "dt" and d_col and c_col:
        cand = parse_utc_datetime(raw[d_col], raw[c_col])
        label_ = "the date and time columns (packet generation time)"
    elif kind == "ts" and t_col:
        cand = to_datetime_flexible(raw[t_col].values)
        label_ = f"the '{TIMESTAMP_COL}' column"
    else:
        continue
    if cand.notna().mean() >= 0.9:
        tt, source = cand, label_
        break
if tt is None:
    print("No usable time. Check DATE_COL, CLOCK_COL and TIMESTAMP_COL. Raw values:")
    cols_ = [c for c in (t_col, d_col, c_col) if c]
    print(raw[cols_].head(8).to_string(index=False))
    sys.exit(1)
print(f"Time taken from {source}")
sig["_t"] = pd.Series(tt.values, index=raw.index)
sig = sig.dropna(subset=["_t"])
if VALID_FROM:
    sig = sig[sig["_t"] >= pd.Timestamp(VALID_FROM)]
if VALID_TO:
    sig = sig[sig["_t"] <= pd.Timestamp(VALID_TO)]
if len(sig) == 0:
    print("Every row is outside the allowed time window or unreadable.")
    sys.exit(1)
centre = sig["_t"].sort_values().iloc[len(sig) // 2]
far = (sig["_t"] - centre).abs() > pd.Timedelta(days=MAX_DAYS_FROM_MEDIAN)
if far.any():
    print(f"Dropped {int(far.sum())} packets dated more than {MAX_DAYS_FROM_MEDIAN} days from the middle "
          "of the data (wrong device clock?).")
    sig = sig[~far]
print("Time range:", sig["_t"].min(), "to", sig["_t"].max())

# numeric vs state columns
cat_signals = [s_ for s_ in STATE_SIGNALS if s_ in sig.columns]
num_signals = [s_ for s_ in found if s_ not in cat_signals]
for s_ in num_signals:
    sig[s_] = pd.to_numeric(sig[s_], errors="coerce")
for s_ in cat_signals:
    sig[s_] = state_text(sig[s_])
    print(f"\nState signal {s_} (top values):")
    print(sig[s_].value_counts(dropna=False).head(8).to_string())

# packets that appear in two files (overlapping days) are kept once
n0 = len(sig)
sig = sig.drop_duplicates().reset_index(drop=True)
if len(sig) < n0:
    print(f"Removed {n0 - len(sig)} duplicate packets (same vehicle, time and values).")

# choose the target signal for motor and for MCU
unused_candidates = set(all_target_names)
banner("TARGETS")
for shown, cands in TARGET_CANDIDATES:
    best, best_cov = None, 0.0
    for c in cands:
        if c in sig.columns:
            cov = float(sig[c].notna().mean())
            print(f"  candidate {c}: present in {100 * cov:.1f}% of rows")
            if cov > best_cov:
                best, best_cov = c, cov
    if best is None or best_cov < 0.01:
        print(f"No usable signal for the {shown} temperature among {cands}.")
        print("Columns in your files:", raw.columns.tolist()[:40])
        sys.exit(1)
    TARGETS.append(best)
    unused_candidates.discard(best)
    print(f"  -> {shown} temperature target: {best}")
num_signals = [s_ for s_ in num_signals if s_ not in unused_candidates]   # a duplicate sensor must not be an input
del raw

sig["_variant"] = sig[DEVICE_COL].map(vmap)
print("\nVehicles and rows per variant:")
print(sig.groupby("_variant")[DEVICE_COL].nunique().rename("vehicles").to_frame().join(
    sig.groupby("_variant").size().rename("rows")))
variant_of = dict(zip(sig[DEVICE_COL], sig["_variant"]))

# =====================================================================
# 4. ALIGN PACKETS ON A REGULAR GRID, BUILD 15-MIN LABELS (per vehicle)
# =====================================================================
def align_vehicle(g, num_cols, cat_cols):
    g = g.sort_values("_t")
    slot = g["_t"].dt.floor(f"{GRID_SEC}s")
    cols = num_cols + cat_cols
    agg = g.groupby(slot)[cols].last()                 # last non-empty value per signal per slot
    grid = pd.date_range(agg.index.min(), agg.index.max(), freq=f"{GRID_SEC}s")
    agg = agg.reindex(grid)
    agg = agg.ffill(limit=FILL_ROWS)                   # carry forward at most FILL_SEC
    for t in TARGETS:                                  # label: value exactly 15 min later on the grid
        agg[f"{t}_future"] = agg[t].shift(-STEPS_AHEAD)
    agg = agg[agg[cols].notna().any(axis=1)]           # drop slots with nothing at all
    agg.index.name = "_t"
    return agg.reset_index()


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


def make_xgb(params):
    return XGBRegressor(n_estimators=XGB_TREES, subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1, **params)


def fit_predict(model, cols, tr, te, target):
    if len(tr) > MAX_TRAIN_ROWS:
        tr = tr.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
    m = clone(model)
    t0 = time.time()
    m.fit(tr[cols], tr[f"{target}_future"])
    return m, m.predict(te[cols]), time.time() - t0


# =====================================================================
# 4a. LSTM helpers
# =====================================================================
def make_sequences(df, cols, target, seq_len, stride):
    """Windows of seq_len consecutive 10-second steps, built per vehicle inside ONE part of the split.
    A window is kept only if it has no time gap. Returns X, label, current temperature at window end."""
    Xs, ys, cs = [], [], []
    ti = cols.index(target)
    for _, g in df.groupby(DEVICE_COL, sort=False):
        g = g.sort_values("_t")
        n = len(g)
        if n < seq_len:
            continue
        t = g["_t"].values.astype("datetime64[s]").astype(np.int64)
        vals = g[cols].values.astype(np.float32)
        y = g[f"{target}_future"].values.astype(np.float32)
        ends = np.arange(seq_len - 1, n, stride)
        ends = ends[(t[ends] - t[ends - (seq_len - 1)]) == (seq_len - 1) * GRID_SEC]
        if len(ends) == 0:
            continue
        Xs.append(np.stack([vals[e - seq_len + 1:e + 1] for e in ends]))
        ys.append(y[ends])
        cs.append(vals[ends, ti])
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs)


def cap(seq, n, seed=RANDOM_SEED):
    if seq is None or len(seq[0]) <= n:
        return seq
    idx = np.random.RandomState(seed).choice(len(seq[0]), n, replace=False)
    return tuple(a[idx] for a in seq)


class LSTMNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm = nn.LSTM(n_in, LSTM_HIDDEN, num_layers=LSTM_LAYERS, batch_first=True,
                            dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(LSTM_HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)


def lstm_predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out.append(model(torch.from_numpy(X[i:i + 4096]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def lstm_train(X, d, epochs, dsd, val=None):
    """val = (Xv, yv, cv). Returns the model (best epoch if val is given), best epoch, best val RMSE."""
    model = LSTMNet(X.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.HuberLoss()
    best_r, best_ep, best_state, bad = np.inf, epochs, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), LSTM_BATCH):
            b = perm[i:i + LSTM_BATCH]
            xb = torch.from_numpy(X[b]).to(DEVICE)
            db = torch.from_numpy(d[b]).to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), db).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val is not None:
            Xv, yv, cv = val
            r = rmse(yv, cv + lstm_predict(model, Xv) * dsd)
            if r < best_r:
                best_r, best_ep, best_state, bad = r, ep, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= LSTM_PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best_r


def lstm_run(train, val, test, cols, target):
    """Train on train windows (epochs chosen on validation), optionally refit on train+validation
    for that many epochs, score once on test windows. The model predicts the 15-min CHANGE."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tr = cap(make_sequences(train, cols, target, LSTM_SEQ_LEN, LSTM_STRIDE), LSTM_MAX_SEQ)
    va = cap(make_sequences(val, cols, target, LSTM_SEQ_LEN, LSTM_STRIDE), LSTM_MAX_SEQ)
    te = cap(make_sequences(test, cols, target, LSTM_SEQ_LEN, LSTM_STRIDE), LSTM_MAX_SEQ)
    if tr is None or va is None or te is None or min(len(tr[0]), len(va[0]), len(te[0])) < 50:
        print(f"  LSTM {target}: too few gap-free windows, skipped.")
        return None
    mu = train[cols].values.astype(np.float32).mean(axis=0)
    sd = train[cols].values.astype(np.float32).std(axis=0)
    sd[sd == 0] = 1.0

    def sc(X):
        return ((X - mu) / sd).astype(np.float32)

    dsd = float(np.std(tr[1] - tr[2])) or 1.0

    def dl(s):
        return ((s[1] - s[2]) / dsd).astype(np.float32)

    t0 = time.time()
    model, best_ep, best_r = lstm_train(sc(tr[0]), dl(tr), LSTM_EPOCHS, dsd,
                                        val=(sc(va[0]), va[1], va[2]))
    if REFIT_ON_TRAIN_VAL:
        Xall = np.concatenate([sc(tr[0]), sc(va[0])])
        dall = np.concatenate([dl(tr), dl(va)])
        model, _, _ = lstm_train(Xall, dall, best_ep, dsd)
    pt = te[2] + lstm_predict(model, sc(te[0])) * dsd
    info = f"seq={LSTM_SEQ_LEN}, hidden={LSTM_HIDDEN}, epochs={best_ep}, test windows={len(te[0])}"
    return best_r, pt, te[1], time.time() - t0, info


# =====================================================================
# 4b. EXPLORATION: summary statistics, distributions, correlations
# =====================================================================
def explore(label, data, sig_cols, present, safe):
    banner(f"EXPLORATION: {label}")
    d = data.copy()
    change_cols = []
    for t in TARGETS:
        d[f"{t}_future_change"] = d[f"{t}_future"] - d[t]       # temperature change in 15 min
        change_cols.append(f"{t}_future_change")
    d_plot = d.sample(EDA_MAX_SAMPLE, random_state=RANDOM_SEED) if len(d) > EDA_MAX_SAMPLE else d

    cols_all = sig_cols + change_cols
    desc = d[cols_all].describe(percentiles=[0.01, 0.5, 0.99]).T
    desc["present_%"] = [round(100 * float(present.get(c, 1.0)), 1) for c in desc.index]
    desc["zeros_%"] = [round(100 * float((d[c] == 0).mean()), 1) for c in desc.index]
    z = (d[cols_all] - d[cols_all].mean()) / d[cols_all].std(ddof=0)
    desc["beyond_5_std_%"] = [round(100 * float((z[c].abs() > 5).mean()), 2) for c in desc.index]
    desc = desc.rename(columns={"1%": "p1", "50%": "median", "99%": "p99"}).round(3)
    show = ["count", "present_%", "min", "p1", "median", "mean", "p99", "max", "std", "zeros_%", "beyond_5_std_%"]
    print(desc[show].to_string())
    desc[show].to_csv(os.path.join(BASE_DIR, f"out_summary_{safe}_{RUN_LABEL}.csv"))

    notes = []
    for c in sig_cols:
        if d[c].nunique() <= 1:
            notes.append(f"{c}: constant (one value only) - carries no information")
        elif (d[c] == 0).mean() > 0.9:
            notes.append(f"{c}: {100 * (d[c] == 0).mean():.0f}% zeros - mostly idle or not reported")
        if float(present.get(c, 1.0)) < 0.8:
            notes.append(f"{c}: present in only {100 * float(present.get(c, 1.0)):.0f}% of aligned rows")
    for t in TARGETS:
        if d[t].min() < -20 or d[t].max() > 200:
            notes.append(f"{t}: range {d[t].min():.1f} to {d[t].max():.1f} C looks implausible - check for error codes")
    print("\nThings to check in the summary:")
    print("\n".join("  - " + n for n in notes) if notes else "  (nothing obvious)")

    n = len(cols_all)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), cols_all):
        ax.axis("on")
        ax.hist(d_plot[c].dropna(), bins=40)
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{label}: distributions (last block = 15-min temperature change)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, f"out_distributions_{safe}_{RUN_LABEL}.png"), dpi=110)
    plt.close(fig)

    corr_cols = sig_cols + (["is_derating"] if "is_derating" in d.columns else []) + change_cols
    corr = d[corr_cols].corr()
    corr.round(3).to_csv(os.path.join(BASE_DIR, f"out_correlation_{safe}_{RUN_LABEL}.csv"))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(corr_cols) + 3), max(5, 0.6 * len(corr_cols) + 2)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_cols, fontsize=8)
    if len(corr_cols) <= 16:
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                v = corr.values[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{label}: correlation between signals")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, f"out_correlation_{safe}_{RUN_LABEL}.png"), dpi=110)
    plt.close(fig)

    for t in TARGETS:
        s_ = corr[f"{t}_future_change"].drop(labels=change_cols, errors="ignore").dropna()
        s_ = s_.reindex(s_.abs().sort_values(ascending=False).index).head(8)
        print(f"\nSignals most correlated with the 15-min change in {t}:")
        print(s_.round(3).to_string())
    pairs = []
    for i, a in enumerate(sig_cols):
        for b in sig_cols[i + 1:]:
            r = corr.loc[a, b]
            if r == r and abs(r) >= HIGH_CORR:
                pairs.append((a, b, round(float(r), 3)))
    print(f"\nSignal pairs with |correlation| >= {HIGH_CORR} (they carry almost the same information):")
    print("\n".join(f"  {a} and {b}: {r}" for a, b, r in pairs) if pairs else "  (none)")

    vid = d[DEVICE_COL].value_counts().index[0]
    one = d[d[DEVICE_COL] == vid].sort_values("_t")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for t in TARGETS:
        ax.plot(one["_t"], one[t], label=t)
    ax.set_title(f"{label}: vehicle {vid} over time")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, f"out_timeseries_{safe}_{RUN_LABEL}.png"), dpi=110)
    plt.close(fig)


# =====================================================================
# 5. ONE MODEL GROUP
# =====================================================================
def run_group(label, frame, variant_as_input):
    banner(f"GROUP: {label}   ({frame[DEVICE_COL].nunique()} vehicles, {len(frame)} raw rows)")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label))
    num_cols = [c for c in num_signals if c in frame.columns]
    cat_cols = [c for c in cat_signals if c in frame.columns]

    # ---- align every vehicle ----
    parts = []
    for dev, g in frame.groupby(DEVICE_COL):
        a = align_vehicle(g, num_cols, cat_cols)
        a[DEVICE_COL] = dev
        parts.append(a)
    data = pd.concat(parts, ignore_index=True)
    data["_variant"] = data[DEVICE_COL].map(variant_of)
    print(f"  after alignment: {len(data)} rows")

    # ---- keep signals this group really has ----
    cover = data[num_cols].notna().mean()
    drop = [c for c in num_cols if cover[c] < MIN_COVERAGE and c not in TARGETS]
    if drop:
        print("  dropped mostly-empty signals:", {c: round(float(cover[c]), 2) for c in drop})
    feats = [c for c in num_cols if c not in drop and c not in DERATING]   # raw derating columns only feed the 0/1 flag

    # ---- state filters ----
    for c in cat_cols:
        data[c] = data[c].fillna("UNKNOWN")
    for s, rule in STATE_FILTERS.items():
        if s not in data.columns:
            continue
        keep = pd.Series(True, index=data.index)
        if "exclude" in rule:
            keep &= ~data[s].isin([str(x).upper() for x in rule["exclude"]])
        if "include" in rule:
            keep &= data[s].isin([str(x).upper() for x in rule["include"]])
        print(f"  state filter on {s}: dropped {int((~keep).sum())} rows")
        data = data[keep]

    # ---- derating flag and one-hot state columns ----
    der = [c for c in DERATING if c in data.columns]
    inputs = [c for c in feats]
    if der:
        data["is_derating"] = (data[der].fillna(0) > 0).any(axis=1).astype(float)
        inputs.append("is_derating")
    onehot = [c for c in CATEGORICAL] + (PACKET_FIELDS if USE_PACKET_FIELDS_AS_INPUTS else [])
    for c in onehot:
        if c not in data.columns:
            continue
        freq = data[c].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        lvl = data[c].where(~data[c].isin(rare), "OTHER")
        dm = pd.get_dummies(lvl, prefix=c, dtype=float)
        data = pd.concat([data, dm], axis=1)
        inputs += list(dm.columns)
    if variant_as_input and data["_variant"].nunique() > 1:
        dm = pd.get_dummies(data["_variant"], prefix="variant", dtype=float)
        data = pd.concat([data, dm], axis=1)
        inputs += list(dm.columns)

    # ---- require every input and both labels ----
    need = [c for c in feats] + [f"{t}_future" for t in TARGETS]
    before = len(data)
    data = data.dropna(subset=need).reset_index(drop=True)
    print(f"  rows with every input and both 15-min labels: {len(data)} of {before}")
    if len(data) < MIN_ROWS_PER_GROUP:
        print("  Skipping: too few usable rows. Fetch longer continuous data for this group.")
        return None
    print(f"  inputs ({len(inputs)}): {inputs}")

    if RUN_EDA:
        explore(label, data, feats, cover, safe)

    # ---- split in time order: train / validation / test ----
    train, val, test = split_three(data)
    print(f"  split in time order (test = latest {TEST_PCT}%; the rest is {TRAIN_SHARE}% train / {VAL_SHARE}% validation): "
          f"train {len(train)} | validation {len(val)} | test {len(test)} rows")
    if min(len(train), len(val), len(test)) < 50:
        print("  Skipping: one of the three parts has fewer than 50 rows. Use more data "
              "or a different split.")
        return None
    trval = pd.concat([train, val]) if REFIT_ON_TRAIN_VAL else train
    val_full, test_full = val, test          # the LSTM needs consecutive rows, so it uses the unsampled parts
    if len(val) > EVAL_SAMPLE:
        val = val.sample(EVAL_SAMPLE, random_state=RANDOM_SEED).sort_index()
        print(f"  validation scored on a random sample of {EVAL_SAMPLE} rows")
    if len(test) > EVAL_SAMPLE:
        test = test.sample(EVAL_SAMPLE, random_state=RANDOM_SEED).sort_index()
        print(f"  test scored on a random sample of {EVAL_SAMPLE} rows")

    rows, grid_rows, preds, importances = [], [], {}, {}
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

        # XGBoost: choose settings on validation, then score once on test
        if RUN_XGBOOST:
            bestx = None
            for params in XGB_GRID:
                _, pv, _ = fit_predict(make_xgb(params), inputs, train, val, t)
                r = rmse(y_val, pv)
                grid_rows.append({"Group": label, "Target": t, "Settings": "XGB " + str(params),
                                  "RMSE_val": round(r, 3)})
                if bestx is None or r < bestx[0]:
                    bestx = (r, params)
            m, pt, secs = fit_predict(make_xgb(bestx[1]), inputs, trval, test, t)
            rows.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "XGBoost",
                         "RMSE_val": round(bestx[0], 3), **score(y_test, pt),
                         "Settings": str(bestx[1]), "Train_sec": round(secs, 1)})
            preds[(t, "XGBoost")] = pt
            models_final[(t, "XGBoost")] = make_xgb(bestx[1])
            importances[(t, "XGB")] = pd.Series(m.feature_importances_, index=inputs)
            print(f"  {t}: best XGBoost settings on validation = {bestx[1]} (validation RMSE {bestx[0]:.3f})")

        # LSTM: scored on gap-free windows from the test part (not stored in preds, rows differ)
        if RUN_LSTM:
            out = lstm_run(train, val_full, test_full, inputs, t)
            if out is not None:
                r_val, pt_l, y_l, secs, info = out
                rows.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "LSTM",
                             "RMSE_val": round(r_val, 3), **score(y_l, pt_l),
                             "Settings": info, "Train_sec": round(secs, 1)})
                print(f"  {t}: LSTM done ({info})")
    res = pd.DataFrame(rows)
    pd.DataFrame(grid_rows).to_csv(os.path.join(BASE_DIR, f"out_rf_validation_{safe}_{RUN_LABEL}.csv"), index=False)

    # ---- which inputs matter ----
    imp_rows = []
    for (t, kind), s_ in importances.items():
        top = s_.reindex(s_.abs().sort_values(ascending=False).index).head(8)
        print(f"\n  Top inputs, {kind} ({'scaled coefficients' if kind == 'LR' else 'importance'}), {t}:")
        print(top.round(3).to_string())
        for name, v in s_.items():
            imp_rows.append({"Group": label, "Target": t, "Model": kind, "Input": name, "Value": round(float(v), 4)})
    pd.DataFrame(imp_rows).to_csv(os.path.join(BASE_DIR, f"out_importance_{safe}_{RUN_LABEL}.csv"), index=False)

    # ---- unseen-vehicle check (Linear Regression, Random Forest, XGBoost; the LSTM is not included) ----
    vehicles = data[DEVICE_COL].unique()
    folds = []
    if RUN_UNSEEN_VEHICLE_CHECK and len(vehicles) >= 2:
        if len(vehicles) <= LOVO_MAX_VEHICLES:
            folds = [[v] for v in vehicles]
        elif len(vehicles) >= 5:
            rng = np.random.RandomState(RANDOM_SEED)
            folds = [list(rng.choice(vehicles, size=max(1, int(len(vehicles) * 0.2)), replace=False))]
    extra = []
    for t in TARGETS:
        for name in [n_ for n_ in ("Linear Regression", "Random Forest", "XGBoost") if (t, n_) in models_final]:
            r = []
            for hold in folds:
                h = data[DEVICE_COL].isin(hold)
                if h.sum() < 50 or (~h).sum() < 50:
                    continue
                _, p, _ = fit_predict(models_final[(t, name)], inputs, data[~h], data[h], t)
                r.append(rmse(data.loc[h, f"{t}_future"], p))
            if r:
                extra.append({"Target": t, "Model": name, "RMSE_unseen_vehicles": round(float(np.mean(r)), 3),
                              "RMSE_worst_unseen_vehicle": round(float(np.max(r)), 3)})
    if extra:
        res = res.merge(pd.DataFrame(extra), on=["Target", "Model"], how="left")
    res = res.round(3)
    res["MAE_below_target"] = res["MAE"] < MAE_TARGET

    # ---- Isolation Forest ----
    if_cols = [c for c in inputs if c != "is_derating"] + [t for t in TARGETS if t not in inputs]
    iso = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION,
                          random_state=RANDOM_SEED, n_jobs=-1).fit(train[if_cols])
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

    # ---- plot: best forecaster per target on the test part (LSTM excluded: it uses its own windows) ----
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(7 * len(TARGETS), 4), squeeze=False)
    for ax, t in zip(axes[0], TARGETS):
        ok_models = [m_ for (tt_, m_) in preds if tt_ == t]
        sub = res[(res["Target"] == t) & (res["Model"].isin(ok_models))].sort_values("RMSE")
        b = sub.iloc[0]["Model"]
        n = min(1000, len(test))
        ax.plot(test[f"{t}_future"].values[:n], label=f"Actual (t+{HORIZON_MIN} min)")
        ax.plot(preds[(t, b)][:n], label=f"Predicted: {b}", alpha=0.8)
        ax.set_title(f"{label} - {t} (test part)")
        ax.set_xlabel("Test sample")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"out_plot_{safe}_{RUN_LABEL}.png"), dpi=120)

    data.to_csv(os.path.join(BASE_DIR, f"out_prepared_{safe}_{RUN_LABEL}.csv"), index=False)
    return res, if_res


# =====================================================================
# 6. RUN: whole 4-wheeler class, and each variant
# =====================================================================
jobs = []
n_variants = sig["_variant"].nunique()
if RUN_CLASS_LEVEL or n_variants == 1:
    jobs.append((f"{CLASS_LABEL}_all", sig, True))
if RUN_VARIANT_LEVEL and n_variants > 1:
    for v, fr in sig.groupby("_variant"):
        jobs.append((f"{CLASS_LABEL}_{v}", fr, False))

all_res, all_if = [], []
for label, frame, variant_in in jobs:
    out = run_group(label, frame, variant_in)
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
print("The LSTM is scored on gap-free windows from the test part (see its Settings column),")
print("so it is judged on a subset of the rows the other models use.")
for (g_, t), sub in results.groupby(["Group", "Target"], sort=False):
    print(f"\n--- {g_} | {t} ---")
    print(sub.sort_values("RMSE").drop(columns=["Run", "Group", "Target"]).to_string(index=False))

banner(f"SUMMARY: best forecaster per group and target (goal: test MAE below {MAE_TARGET})")
summ = []
for (g_, t), sub in results.groupby(["Group", "Target"], sort=False):
    base = sub[sub["Model"] == "Persistence baseline"].iloc[0]
    best = sub[sub["Model"] != "Persistence baseline"].sort_values("RMSE").iloc[0]
    summ.append({"Group": g_, "Target": t, "Best model": best["Model"], "Test MAE": best["MAE"],
                 "Test RMSE": best["RMSE"], "Baseline MAE": base["MAE"],
                 "MAE below target": bool(best["MAE"] < MAE_TARGET),
                 "Beats baseline": bool(best["RMSE"] < base["RMSE"])})
summ = pd.DataFrame(summ)
print(summ.to_string(index=False))
summ.to_csv(os.path.join(BASE_DIR, f"out_summary_best_{RUN_LABEL}.csv"), index=False)
print("A result counts only when BOTH 'MAE below target' and 'Beats baseline' are True.")

banner("ISOLATION FOREST (unsupervised: compare flagged vs normal rows, no RMSE)")
print(if_all.to_string(index=False))
print("\nA forecaster counts only if its RMSE is below the Persistence baseline.")
print("If 'Flagged % (test)' is far above the setting IF_CONTAMINATION, the test period looks")
print("different from the training period; that is not a fault rate.")
banner("DONE - files starting with out_ are in the folder of this script")
plt.show()
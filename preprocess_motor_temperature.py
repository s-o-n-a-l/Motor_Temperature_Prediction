# =====================================================================
# STEP 1 OF 2 - PRE-PROCESSING
# 4-wheeler buses (12M, 6S, 3S ...): motor + MCU temperature, 15 minutes ahead
#
# What this file does (no model is trained here):
#   1. reads the vehicle list and picks the vehicles / variants / groups
#   2. reads the big monthly CSV files chunk by chunk and reduces them while reading
#   3. summer statistics: which vehicle is hottest in each month (March-May), from ALL vehicles
#   4. puts every vehicle on a regular 10-second grid and builds the 15-minute-ahead labels
#   5. builds the inputs (derating flag, one-hot state columns, variant), cleans, explores
#      (summary, distributions, correlation, time series) and SAVES one model-ready table per group
#
# Output (in the "prepared" folder): prepared_<RUN_LABEL>_<group>.pkl, a small CSV sample of each,
# and prepared_index_<RUN_LABEL>.json, which tells step 2 what was prepared.
#
# Next: run  02_train_validate_test.py   (splits in time order, trains, validates, tests)
#
# Run (VS Code terminal, .venv active):  python3 01_preprocess_data.py
# Write your own version for your project and follow your institution's rules on disclosing AI assistance.
# =====================================================================

import os
import re
import sys
import json
import ast
import pickle
import hashlib
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # save PNG files only; no Tk window, no crash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =====================================================================
# SETTINGS  (edit this block only)
# =====================================================================
RUN_LABEL = "month5"                              # change for each new batch of months; use THE SAME text in step 2
CSV_FILES = [                                         # the monthly CSV files; all are joined
Path("/home/arvr-lab-16/Downloads/motor temperature /month5 _10device_ids/37bdf528-3a79-458d-87b0-735569882e94.csv")
]
CSV_FOLDER = None                                     # or a folder: every .csv inside is used (when CSV_FILES is empty)
CHUNK_ROWS = 300000                                   # files are read this many rows at a time (smaller = less memory)
VEHICLE_LIST_CSV = Path("/home/arvr-lab-16/Downloads/motor temperature /CAN Signals Info - updated.xlsx")   # vehicle list (.xlsx or .csv)
VEHICLE_LIST_SHEET = 0                                # sheet name or number (only used for .xlsx)
VEHICLE_LIST_DEVICE_COL = "device_id"
VEHICLE_LIST_VARIANT_COL = "variant"

VARIANTS = ["12M","9M","55T"]                        # variants to load (text must match the vehicle list); None = every variant in the list
CLASS_LABEL = "ALL"                                   # name used in the results (ALL_all for everything, ALL_<group> per group)

# Optional: merge variants into bigger groups, e.g. {"12M": "BUS", "3S": "3W", "4S": "3W"}.
# Variants not listed here are their own group.
GROUP_MAP = {}
GROUPS_TO_RUN = None                                  # None = all groups, or e.g. ["BUS"]
RUN_CLASS_LEVEL = True                                # prepare one table for all chosen groups together (ALL_all)
RUN_GROUP_LEVEL = True                                # plus one table per group (skipped when there is only one group)

# Memory control for very large files (e.g. several 10 GB CSVs)
VEHICLE_SAMPLE_FRACTION = None                        # e.g. 0.25 = models use about 25 % of the vehicles (fixed choice by device_id)
LOAD_MONTHS = None                                    # e.g. [3, 4, 5] = models use only packets of these months
PRE_AGGREGATE = True                                  # while loading keep one row per vehicle and GRID_SEC slot (same result, far fewer rows)
USE_CACHE = True                                      # save the reduced data after the first load; later runs skip reading the CSV files
CACHE_FOLDER = None                                   # None = a folder "cache" next to this script
PREPARED_FOLDER = None                                # None = a folder "prepared" next to this script (step 2 reads from here)
# The summer (hottest vehicle) tables always use ALL selected vehicles, whatever the sample / month settings.

# Summer season for the "hottest vehicle" tables (months as numbers)
SUMMER_MONTHS = [5]                             # March, April, May
FOCUS_MONTH = 1                                      # month printed in detail (hottest vehicles); every summer month is saved to a CSV
SUMMER_MIN_READINGS = 50                              # a vehicle needs at least this many readings to be listed
SUMMER_TOP_N = 10

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
TREAT_ZERO_TEMP_AS_MISSING = True                     # a reading of exactly 0 is a default / start-up value, not a temperature

# Number inputs (spelling as in your data). Names not found are skipped with a message.
# Left out on purpose: Vehicle_DTE_Km and MCU_Odometer_Val (they identify a vehicle or
# just grow with time, so a model can learn the vehicle instead of the physics).
NUMERIC_FEATURES = [
    "MCU_HighPowerVoltage", "MCU_HighPowerCurrent", "MCU_Motor_Speed",
    "Torque_Command", "MCU_Motor_Torque_Estimated", "Traction_Torque_Avaliable",
    "Ambient_Temprature","MCU_DTC_1","MCU_DTC_2","MCU_State",      # dropped if absent or mostly empty
    "Pump_Speed", "Pump_Voltage", "Pump_Current", "TCS_Pump_Temperature",   # cooling signals, same rule
]
DERATING = []   # combined into one 0/1 flag
CATEGORICAL = ["ignition","VCU_Flag"]   # state signals -> 0/1 columns,MCU_Drive_Mode:0-neutral,1-eco,2-boost,3-reverse,4-fault
PACKET_FIELDS = ["packet_type", "packet_status"]      # printed for inspection; see next line
USE_PACKET_FIELDS_AS_INPUTS = False                   # False = use them for cleaning only (recommended)

# Rows to drop or keep by state. Values are compared as upper-case text.
#   "ignition": {"include": ["1"]},   "MCU_Drive_Mode": {"exclude": ["0"]},   "packet_status": {"include": ["L"]}
STATE_FILTERS = {}

MAX_DAYS_FROM_MEDIAN = 200               # packets dated farther than this from the middle of the data are dropped
VALID_FROM, VALID_TO = None, None        # optional hard limits, e.g. "2026-01-01", "2026-06-30"
PREFER_DATE_TIME = True                  # time = packet generation time (date + time columns) before 'timestamp'

# Time handling
GRID_SEC = 10            # packets are put on a regular 10-second grid
FILL_SEC = 60            # a signal is carried forward at most this long
HORIZON_MIN = 15         # predict this many minutes ahead
MIN_COVERAGE = 0.3       # a signal present in less than 30% of a group's rows is dropped
MIN_ROWS_PER_GROUP = 300

# Sliding-window features: statistics of the LAST few minutes of a signal, per vehicle, on the 10-second grid.
# Only past data is used (a trailing window ends at the current row), so nothing from the future leaks in.
USE_SLIDING_WINDOWS = True
WINDOW_MINUTES = [1, 5, 15]                           # window lengths in minutes
WINDOW_STATS = ["mean", "max"]                        # any of: "mean", "max", "min", "std"
WINDOW_ON_TARGETS = True                              # also for the motor and MCU temperature themselves
WINDOW_SIGNALS = ["MCU_HighPowerCurrent", "MCU_HighPowerVoltage", "MCU_Motor_Speed", "Torque_Command"]   # other signals (skipped if absent / dropped)
WINDOW_MINUS_MEAN = True                              # also "signal now minus its window mean" (rising / falling indicator)


# Exploration (summary statistics, distributions, correlations)
RUN_EDA = True
EDA_MAX_SAMPLE = 200000
HIGH_CORR = 0.95
SAVE_SAMPLE_CSV_ROWS = 20000             # a small CSV of the prepared table to look at in Excel (0 = none)
RANDOM_SEED = 42

# =====================================================================
# Below this line: no need to edit
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPARED_DIR = Path(PREPARED_FOLDER) if PREPARED_FOLDER else Path(BASE_DIR) / "prepared"
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


def parse_utc_datetime(date_s, time_s, state):
    """Combine separate date and time columns. Tries common formats (05012026 / 081530,
    2026-01-05, 05/01/2026, 12-hour clock ...), strips 'UTC'/'Z', restores lost leading zeros.
    The format is detected on the first call and kept in `state`, so later chunks reuse it."""
    d = date_s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    t = time_s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    d = d.str.replace(r"\s*(UTC|Z)$", "", regex=True)
    t = t.str.replace(r"\s*(UTC|Z)$", "", regex=True)
    if d.str.fullmatch(r"\d+").mean() > 0.9:
        d = d.str.zfill(state.setdefault("dpad", 8 if d.str.len().max() >= 7 else 6))
    if t.str.fullmatch(r"\d+").mean() > 0.9:
        t = t.str.zfill(6)
    both_s = d + " " + t
    fmt0 = state.get("fmt")
    if fmt0:
        out = pd.to_datetime(both_s, format=fmt0, errors="coerce")
        if out.notna().mean() >= 0.9:
            return out
    date_fmts = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d%m%y",
                 "%d%m%Y", "%y%m%d", "%Y%m%d", "%d-%m-%y", "%d/%m/%y", "%m/%d/%y"]
    time_fmts = ["%H:%M:%S", "%H%M%S", "%I:%M:%S %p", "%H:%M:%S.%f", "%H:%M"]
    both = both_s.dropna()
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
    state["fmt"] = best
    print(f"  date/time format detected: '{best}'  ({100 * best_rate:.0f}% of sampled rows parse)")
    return pd.to_datetime(both_s, format=best, errors="coerce")


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
want = {norm(v) for v in VARIANTS} if VARIANTS else set(vl["_var"].map(norm))
vl_sel = vl[vl["_var"].map(norm).isin(want)]
if vl_sel.empty:
    print(f"\nNone of {VARIANTS} appear in the list above. Fix VARIANTS so the text matches.")
    sys.exit(1)
vmap = dict(zip(vl_sel["_id"], vl_sel["_var"].str.upper()))

# groups: GROUP_MAP merges variants; variants not listed there are their own group
group_map = {str(k).strip().upper(): str(v).strip() for k, v in GROUP_MAP.items()}
unknown_keys = [k for k in group_map if k not in set(vmap.values())]
if unknown_keys:
    print(f"Note: GROUP_MAP names {unknown_keys}, which are not among the selected variants (ignored).")
group_of_variant = {v: group_map.get(v, v) for v in set(vmap.values())}
gmap = {dev: group_of_variant[v] for dev, v in vmap.items()}          # device_id -> group
print("\nGroups (variants inside):")
for g_ in sorted(set(gmap.values())):
    members = sorted(v for v, gg in group_of_variant.items() if gg == g_)
    print(f"  {g_}: {members}  ({sum(1 for x in gmap.values() if x == g_)} vehicles)")
if GROUPS_TO_RUN:
    wanted_groups = {str(g).strip().upper() for g in GROUPS_TO_RUN}
    gmap = {d: g for d, g in gmap.items() if g.upper() in wanted_groups}
    if not gmap:
        print(f"\nNone of GROUPS_TO_RUN={GROUPS_TO_RUN} matches a group above. Fix the names.")
        sys.exit(1)
    vmap = {d: v for d, v in vmap.items() if d in gmap}
    print(f"Only these groups are run: {sorted(set(gmap.values()))}")
keep_ids = set(vmap)
print(f"\nVehicles selected: {len(keep_ids)}")

# =====================================================================
# 3. LOAD THE CSV FILES
#    Every chunk is reduced BEFORE it is kept, so 10 GB files do not fill the memory:
#    selected vehicles -> needed columns -> time -> summer statistics (all selected vehicles)
#    -> optional month filter / vehicle sample -> one row per vehicle and GRID_SEC slot
#    -> compact types (float32, state signals as small codes, device_id as category).
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
wanted_all = list(dict.fromkeys(all_target_names + NUMERIC_FEATURES + STATE_SIGNALS))
num_names = [s_ for s_ in wanted_all if s_ not in STATE_SIGNALS]
cat_ids = sorted(keep_ids)                                   # fixed categories for the device_id column
sample_cut = None if not VEHICLE_SAMPLE_FRACTION else int(round(10000 * VEHICLE_SAMPLE_FRACTION))
if sample_cut is not None:
    print(f"Vehicle sample: about {100 * VEHICLE_SAMPLE_FRACTION:.0f}% of the vehicles are used for the models "
          "(fixed choice by device_id; the summer table still uses all selected vehicles).")
if LOAD_MONTHS:
    print(f"Only packets of months {LOAD_MONTHS} are kept for the models.")

dt_state = {}                                                # date/time format, detected on the first chunk
state_codes = {s_: {} for s_ in STATE_SIGNALS}               # state value -> small code (grows while loading)
found_names, zero_count = set(), {n: 0 for n in all_target_names}
summer_parts, frames = [], []
total_rows = kept_rows = rows_after = bad_time_rows = 0
first_header = None


def state_to_codes(name, series):
    txt = state_text(series)
    codes = state_codes[name]
    for v in txt.dropna().unique():
        if v not in codes:
            codes[v] = len(codes)
    return txt.map(codes).astype("float32")                  # NaN stays NaN


def chunk_times(chunk, cm):
    d_col, c_col, t_col = cm.get(norm(DATE_COL)), cm.get(norm(CLOCK_COL)), cm.get(norm(TIMESTAMP_COL))
    for kind in (("dt", "ts") if PREFER_DATE_TIME else ("ts", "dt")):
        if kind == "dt" and d_col and c_col:
            cand = parse_utc_datetime(chunk[d_col], chunk[c_col], dt_state)
        elif kind == "ts" and t_col:
            cand = to_datetime_flexible(chunk[t_col].values)
        else:
            continue
        if cand.notna().mean() >= 0.9:
            return cand.values
    return None


cache_key = hashlib.md5(repr((
    [(f.name, f.stat().st_size) for f in files], sorted(keep_ids), VEHICLE_SAMPLE_FRACTION, LOAD_MONTHS,
    PRE_AGGREGATE, SUMMER_MONTHS, "summer-by-month-v2", VALID_FROM, VALID_TO, GRID_SEC, TREAT_ZERO_TEMP_AS_MISSING, wanted_all,
    STATE_SIGNALS, PREFER_DATE_TIME, DATE_COL, CLOCK_COL, TIMESTAMP_COL, DEVICE_COL)).encode()).hexdigest()[:12]
cache_dir = Path(CACHE_FOLDER) if CACHE_FOLDER else Path(BASE_DIR) / "cache"
cache_path = cache_dir / f"loaded_{RUN_LABEL}_{cache_key}.pkl"
files_to_read = files
if USE_CACHE and cache_path.exists():
    print(f"Reduced data found in the cache, so the CSV files are not read again:\n  {cache_path}")
    with open(cache_path, "rb") as fh:
        c_ = pickle.load(fh)
    frames, summer_parts, state_codes = c_["frames"], c_["summer_parts"], c_["state_codes"]
    found_names, zero_count, first_header = c_["found_names"], c_["zero_count"], c_["first_header"]
    total_rows, kept_rows, rows_after, bad_time_rows = c_["counts"]
    files_to_read = []

for f in files_to_read:
    try:
        header = pd.read_csv(f, nrows=0).columns
    except UnicodeDecodeError:
        print(f"{f.name} is not plain UTF-8 text. Run:  file \"{f}\"  to see what it is.")
        sys.exit(1)
    cm = {norm(c): c for c in header}
    if first_header is None:
        first_header = list(header)
    if norm(DEVICE_COL) not in cm:
        print(f"{f.name}: no '{DEVICE_COL}' column. Columns: {list(header)[:30]}")
        sys.exit(1)
    dev_col = cm[norm(DEVICE_COL)]
    use = [cm[norm(n)] for n in [DEVICE_COL, TIMESTAMP_COL, DATE_COL, CLOCK_COL] + wanted_all if norm(n) in cm]
    file_missing = [s_ for s_ in wanted_all if norm(s_) not in cm]
    can_col = cm.get(norm("can_data")) if file_missing else None
    if can_col:
        use.append(can_col)
    use = list(dict.fromkeys(use))
    found_names.update(s_ for s_ in wanted_all if norm(s_) in cm)
    keymap = None
    n_file = k_file = r_file = 0
    for chunk in pd.read_csv(f, usecols=use, chunksize=CHUNK_ROWS, low_memory=False):
        n_file += len(chunk)
        dev = clean_id(chunk[dev_col])
        ok = dev.isin(keep_ids).values
        if not ok.any():
            continue
        chunk = chunk.loc[ok].reset_index(drop=True)
        dev = dev[ok].reset_index(drop=True)
        k_file += len(chunk)

        s = pd.DataFrame({DEVICE_COL: dev})
        for name in wanted_all:
            if norm(name) in cm:
                s[name] = chunk[cm[norm(name)]].values
        if can_col:                                          # signals missing as columns: read them from can_data
            parsed = chunk[can_col].map(parse_cell)
            if keymap is None:
                keymap = {}
                for dct in parsed.head(2000):
                    for k in dct.keys():
                        keymap.setdefault(norm(k), k)
            for name in file_missing:
                if norm(name) in keymap:
                    k = keymap[norm(name)]
                    s[name] = parsed.map(lambda dct, k=k: dct.get(k)).values
                    found_names.add(name)

        tt = chunk_times(chunk, cm)
        del chunk
        if tt is None:
            bad_time_rows += len(s)
            continue
        s["_t"] = tt
        s = s[s["_t"].notna()]
        if VALID_FROM:
            s = s[s["_t"] >= pd.Timestamp(VALID_FROM)]
        if VALID_TO:
            s = s[s["_t"] <= pd.Timestamp(VALID_TO)]
        if s.empty:
            continue

        for name in num_names:
            if name in s.columns:
                s[name] = pd.to_numeric(s[name], errors="coerce").astype("float32")
        if TREAT_ZERO_TEMP_AS_MISSING:                       # 0 = default / start-up value, not a temperature
            for name in all_target_names:
                if name in s.columns:
                    z = (s[name] == 0)
                    zero_count[name] += int(z.sum())
                    s[name] = s[name].mask(z)

        # summer statistics from ALL selected vehicles (max, sum, count per vehicle)
        in_summer = s["_t"].dt.month.isin(SUMMER_MONTHS)
        t_cols = [n for n in all_target_names if n in s.columns]
        if in_summer.any() and t_cols:
            sub = s.loc[in_summer, [DEVICE_COL] + t_cols].astype({n: "float64" for n in t_cols})
            sub["_m"] = s.loc[in_summer, "_t"].dt.month.values
            g = sub.groupby([DEVICE_COL, "_m"], sort=False)
            parts_ = {}
            for n in t_cols:
                parts_[f"{n}|max"] = g[n].max()
                parts_[f"{n}|sum"] = g[n].sum()
                parts_[f"{n}|cnt"] = g[n].count()
            summer_parts.append(pd.concat(parts_, axis=1))

        # reduce what is kept for the models
        if LOAD_MONTHS:
            s = s[s["_t"].dt.month.isin(LOAD_MONTHS)]
        if sample_cut is not None and len(s):
            h = pd.util.hash_pandas_object(s[DEVICE_COL], index=False).values % 10000
            s = s[h < sample_cut]
        if s.empty:
            continue
        for name in STATE_SIGNALS:
            if name in s.columns:
                s[name] = state_to_codes(name, s[name])
        if PRE_AGGREGATE:                                    # same result as the later grid step, far fewer rows
            s = s.sort_values("_t", kind="stable")
            s["_t"] = s["_t"].dt.floor(f"{GRID_SEC}s")
            s = s.groupby([DEVICE_COL, "_t"], sort=False).last().reset_index()
        s[DEVICE_COL] = pd.Categorical(s[DEVICE_COL], categories=cat_ids)
        r_file += len(s)
        frames.append(s)
    total_rows += n_file
    kept_rows += k_file
    rows_after += r_file
    print(f"  {f.name}: {n_file} rows read, {k_file} belong to the selected vehicles, {r_file} kept for the models")
    if keymap is not None:
        print("    (some signals were read from the can_data column)")

if USE_CACHE and files_to_read and frames:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"frames": frames, "summer_parts": summer_parts, "state_codes": state_codes,
                         "found_names": found_names, "zero_count": zero_count, "first_header": first_header,
                         "counts": (total_rows, kept_rows, rows_after, bad_time_rows)},
                        fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Reduced data saved to the cache (next runs with the same loading settings skip the CSV files):\n  {cache_path}")
    except Exception as e:
        print(f"  (the cache could not be written: {type(e).__name__}: {str(e)[:100]})")

if bad_time_rows:
    print(f"  {bad_time_rows} rows had no readable time and were dropped.")
if not frames and not summer_parts:
    print("No telemetry row belongs to the selected vehicles (or none has a readable time).")
    print("Check that device_id looks the same in the telemetry files and in the vehicle list, and the DATE/TIME columns.")
    sys.exit(1)
print(f"Together: {kept_rows} of {total_rows} rows belong to the selected vehicles; {rows_after} kept for the models")
if TREAT_ZERO_TEMP_AS_MISSING:
    print("Temperature readings of exactly 0 treated as missing:",
          {n: c for n, c in zero_count.items() if n in found_names})
for s_ in wanted_all:
    if s_ not in found_names:
        print(f"  NOT FOUND: {s_}")

if not frames:
    print("Nothing is left for the models (check LOAD_MONTHS, VALID_FROM / VALID_TO and VEHICLE_SAMPLE_FRACTION).")
    sys.exit(1)
sig = pd.concat(frames, ignore_index=True)
del frames
for n in num_names:
    if n in sig.columns:
        sig[n] = sig[n].astype("float32")
print("Time range:", sig["_t"].min(), "to", sig["_t"].max())

# packets with a wrong device clock: far from the (sampled) middle of the data
step = max(1, len(sig) // 1000000)
centre = pd.Series(sig["_t"].values[::step]).sort_values().iloc[len(sig[::step]) // 2]
far = ((sig["_t"] - centre).abs() > pd.Timedelta(days=MAX_DAYS_FROM_MEDIAN)).values
if far.any():
    print(f"Dropped {int(far.sum())} packets dated more than {MAX_DAYS_FROM_MEDIAN} days from the middle "
          "of the data (wrong device clock?).")
    sig = sig[~far].reset_index(drop=True)
    print("Time range now:", sig["_t"].min(), "to", sig["_t"].max())
del far

# state signals are stored as codes: show their values, keep the code -> text lists
cat_signals = [s_ for s_ in STATE_SIGNALS if s_ in sig.columns]
STATE_LABELS = {}
for s_ in cat_signals:
    STATE_LABELS[s_] = sorted(state_codes[s_], key=state_codes[s_].get)
    vc = sig[s_].value_counts(dropna=False).head(8)
    names = [STATE_LABELS[s_][int(i)] if i == i else "NaN" for i in vc.index]
    print(f"\nState signal {s_} (top values):")
    print(pd.Series(vc.values, index=names, name="count").to_string())
found = [s_ for s_ in wanted_all if s_ in sig.columns]
num_signals = [s_ for s_ in found if s_ not in cat_signals]


def codes_to_labels(codes, labels):
    """State codes (float, NaN = missing) -> categorical text; missing becomes 'UNKNOWN'."""
    codes = np.asarray(codes, dtype="float64")
    arr = np.where(np.isnan(codes), len(labels), np.nan_to_num(codes, nan=0.0)).astype("int32")
    return pd.Categorical.from_codes(arr, categories=list(labels) + ["UNKNOWN"])


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
        print("Columns in your files:", (first_header or [])[:40])
        sys.exit(1)
    TARGETS.append(best)
    unused_candidates.discard(best)
    print(f"  -> {shown} temperature target: {best}")
num_signals = [s_ for s_ in num_signals if s_ not in unused_candidates]   # a duplicate sensor must not be an input

# vehicles, variants and groups actually present in the data
rows_by_dev = sig[DEVICE_COL].value_counts()
rows_by_dev = rows_by_dev[rows_by_dev > 0]
dev_table = pd.DataFrame({"rows": rows_by_dev.values}, index=[str(d) for d in rows_by_dev.index])
dev_table["variant"] = [vmap[d] for d in dev_table.index]
dev_table["group"] = [gmap[d] for d in dev_table.index]
print("\nVehicles and rows per variant (models):")
print(dev_table.groupby("variant").agg(vehicles=("rows", "size"), rows=("rows", "sum")))
print("\nVehicles and rows per group (models):")
print(dev_table.groupby("group").agg(vehicles=("rows", "size"), rows=("rows", "sum")))
variant_of = vmap
group_devices = {g_: sorted(dev_table.index[dev_table["group"] == g_]) for g_ in sorted(dev_table["group"].unique())}

# =====================================================================
# 3b. SUMMER (MARCH-MAY): WHICH VEHICLE / VARIANT RUNS HOTTEST
#     computed while loading, from ALL selected vehicles (not only the model sample)
# =====================================================================
banner(f"SUMMER (months {SUMMER_MONTHS}): HOTTEST VEHICLES")
summer_tab, month_tab = pd.DataFrame(), pd.DataFrame()
if not summer_parts:
    print("No readings in the summer months were found.")
else:
    allp = pd.concat(summer_parts)
    del summer_parts
    agg_how = {c: ("max" if c.endswith("|max") else "sum") for c in allp.columns}
    by_month = allp.groupby(level=[0, 1]).agg(agg_how)          # per vehicle and month
    overall = by_month.groupby(level=0).agg(agg_how)             # per vehicle, all summer months together
    del allp

    def hottest_table(stats, t):
        cnt = stats[f"{t}|cnt"]
        tab = pd.DataFrame({"max_temp": stats[f"{t}|max"],
                            "mean_temp": stats[f"{t}|sum"] / cnt.replace(0, np.nan),
                            "readings": cnt})
        tab = tab[tab["readings"] >= SUMMER_MIN_READINGS].reset_index()
        tab["_variant"] = tab[DEVICE_COL].map(vmap)
        tab["_group"] = tab[DEVICE_COL].map(gmap)
        tab.insert(0, "Target", t)
        return tab

    parts_all, parts_month = [], []
    for t in TARGETS:
        if f"{t}|max" not in overall.columns:
            continue
        parts_all.append(hottest_table(overall, t))
        mt = hottest_table(by_month, t).rename(columns={"_m": "Month"})
        parts_month.append(mt)
    if parts_all:
        summer_tab = pd.concat(parts_all, ignore_index=True)
        month_tab = pd.concat(parts_month, ignore_index=True)
        print(f"Vehicles with summer readings: {overall.shape[0]}")

if summer_tab.empty:
    print("No vehicle has enough summer readings. Lower SUMMER_MIN_READINGS or check SUMMER_MONTHS.")
else:
    cols_out = ["Target", DEVICE_COL, "_variant", "_group", "max_temp", "mean_temp", "readings"]
    summer_tab = summer_tab[cols_out].sort_values(["Target", "max_temp"], ascending=[True, False])
    month_tab = month_tab[["Target", "Month"] + cols_out[1:]].sort_values(
        ["Target", "Month", "max_temp"], ascending=[True, True, False])
    summer_tab.round(1).to_csv(os.path.join(BASE_DIR, f"out_summer_hottest_{RUN_LABEL}.csv"), index=False)
    month_tab.round(1).to_csv(os.path.join(BASE_DIR, f"out_hottest_by_month_{RUN_LABEL}.csv"), index=False)

    def show(sub, title):
        print(f"\n{title}")
        print(sub.head(SUMMER_TOP_N).round(1).to_string(index=False))
        print("  hottest vehicle in each variant:")
        print(sub.groupby("_variant").head(1).round(1).to_string(index=False))
        if (sub["_group"] != sub["_variant"]).any():
            print("  hottest vehicle in each group:")
            print(sub.groupby("_group").head(1).round(1).to_string(index=False))

    for t in TARGETS:
        mt = month_tab[(month_tab["Target"] == t) & (month_tab["Month"] == FOCUS_MONTH)]
        if mt.empty:
            print(f"\nNo {t} readings in month {FOCUS_MONTH}.")
        else:
            show(mt, f"Top {SUMMER_TOP_N} vehicles by max {t} in month {FOCUS_MONTH}:")
        mm = month_tab[month_tab["Target"] == t]
        print(f"\nHottest vehicle in each month ({t}):")
        print(mm.groupby("Month").head(1).round(1).to_string(index=False))
        show(summer_tab[summer_tab["Target"] == t], f"Top {SUMMER_TOP_N} vehicles by max {t}, all months {SUMMER_MONTHS} together:")
        sub = summer_tab[summer_tab["Target"] == t]
        print(f"\nVariant comparison ({t}, months {SUMMER_MONTHS}):")
        print(sub.groupby("_variant").agg(vehicles=(DEVICE_COL, "nunique"),
                                          hottest_max=("max_temp", "max"),
                                          mean_of_vehicle_mean=("mean_temp", "mean")).round(1).to_string())
# =====================================================================
# 4. ALIGN PACKETS ON A REGULAR GRID, BUILD 15-MIN LABELS (per vehicle)
# =====================================================================

def window_spec(num_cols):
    if not USE_SLIDING_WINDOWS:
        return []
    names = list(dict.fromkeys((TARGETS if WINDOW_ON_TARGETS else []) + list(WINDOW_SIGNALS)))
    spec = []
    for sgl in names:
        if sgl not in num_cols:
            continue
        for m in WINDOW_MINUTES:
            for st in WINDOW_STATS:
                spec.append((f"{sgl}_{st}_{m}m", sgl, st, m))
            if WINDOW_MINUS_MEAN:
                spec.append((f"{sgl}_minus_mean_{m}m", sgl, "minus_mean", m))
    return spec

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
        spec = window_spec(num_cols)
    if spec:                                           # trailing windows on the regular grid (past + current row only)
        new, rolls = {}, {}
        for name, sgl, st, m in spec:
            n = max(1, int(round(m * 60 / GRID_SEC)))
            if (sgl, n) not in rolls:
                rolls[(sgl, n)] = agg[sgl].rolling(n, min_periods=1)
            r = rolls[(sgl, n)]
            if st == "mean":
                new[name] = r.mean()
            elif st == "max":
                new[name] = r.max()
            elif st == "min":
                new[name] = r.min()
            elif st == "std":
                new[name] = r.std(ddof=0)
            elif st == "minus_mean":
                new[name] = agg[sgl] - r.mean()
    agg = agg[agg[cols].notna().any(axis=1)]           # drop slots with nothing at all
    agg.index.name = "_t"
    return agg.reset_index()


# =====================================================================
# 4b. EXPLORATION: summary statistics, distributions, correlations
# =====================================================================
def explore(label, data, sig_cols, present, safe):
    banner(f"EXPLORATION: {label}")
    d = data.sample(EDA_MAX_SAMPLE, random_state=RANDOM_SEED) if len(data) > EDA_MAX_SAMPLE else data.copy()
    if len(data) > EDA_MAX_SAMPLE:
        print(f"  (statistics and plots below use a random sample of {EDA_MAX_SAMPLE} of {len(data)} rows)")
    change_cols = []
    for t in TARGETS:
        d[f"{t}_future_change"] = d[f"{t}_future"] - d[t]       # temperature change in 15 min
        change_cols.append(f"{t}_future_change")
    d_plot = d

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

    corr_cols = sig_cols + change_cols
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

    vid = data[DEVICE_COL].value_counts().index[0]
    one = data[data[DEVICE_COL] == vid].sort_values("_t")
    one = one.iloc[::max(1, len(one) // 20000)]
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
# 5. PREPARE ONE GROUP (align, labels, inputs, exploration, save)
# =====================================================================
def prepare_group(label, frame, variant_as_input):
    banner(f"GROUP: {label}   ({frame[DEVICE_COL].nunique()} vehicles, {len(frame)} raw rows)")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label))
    num_cols = [c for c in num_signals if c in frame.columns]
    cat_cols = [c for c in cat_signals if c in frame.columns]

    # ---- align every vehicle ----
    parts = []
    for dev, g in frame.groupby(DEVICE_COL, observed=True):
        a = align_vehicle(g, num_cols, cat_cols)
        a[DEVICE_COL] = dev
        parts.append(a)
    data = pd.concat(parts, ignore_index=True)
    del parts
    data["_variant"] = data[DEVICE_COL].map(variant_of).astype("category")
    for c in cat_cols:                                 # state codes -> text labels (missing = UNKNOWN)
        data[c] = codes_to_labels(data[c].values, STATE_LABELS[c])
    print(f"  after alignment: {len(data)} rows")

    # ---- keep signals this group really has ----
    cover = data[num_cols].notna().mean()
    drop = [c for c in num_cols if cover[c] < MIN_COVERAGE and c not in TARGETS]
    if drop:
        print("  dropped mostly-empty signals:", {c: round(float(cover[c]), 2) for c in drop})
    feats = [c for c in num_cols if c not in drop ]   # raw derating columns only feed the 0/1 flag

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
    #der = [c for c in DERATING if c in data.columns]
    inputs = [c for c in feats]
    wmap = {name: base for name, base, _, _ in window_spec(num_cols)}
    win_feats = [c for c, b in wmap.items() if b not in drop and c in data.columns]
    inputs += win_feats
    if win_feats:
        print(f"  sliding-window features added: {len(win_feats)}")
    #if der:
    #    data["is_derating"] = (data[der].fillna(0) > 0).any(axis=1).astype(float)
     #   inputs.append("is_derating")
    onehot = [c for c in CATEGORICAL] + (PACKET_FIELDS if USE_PACKET_FIELDS_AS_INPUTS else [])
    for c in onehot:
        if c not in data.columns:
            continue
        lvl_obj = data[c].astype(object)
        freq = lvl_obj.value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        lvl = lvl_obj.where(~lvl_obj.isin(rare), "OTHER")
        dm = pd.get_dummies(lvl, prefix=c, dtype="float32")
        data = pd.concat([data, dm], axis=1)
        inputs += list(dm.columns)
    if variant_as_input and data["_variant"].nunique() > 1:
        dm = pd.get_dummies(data["_variant"], prefix="variant", dtype="float32")
        data = pd.concat([data, dm], axis=1)
        inputs += list(dm.columns)

    # ---- require every input and both labels ----
    need = [c for c in feats] + win_feats + [f"{t}_future" for t in TARGETS]
    before = len(data)
    data = data.dropna(subset=need).reset_index(drop=True)
    print(f"  rows with every input and both 15-min labels: {len(data)} of {before}")
    if len(data) < MIN_ROWS_PER_GROUP:
        print("  Skipping: too few usable rows. Fetch longer continuous data for this group.")
        return None

    # state columns that never change carry no information: drop them from the inputs
    const_inputs = [c for c in inputs if data[c].nunique() <= 1]
    if const_inputs:
        print("  constant inputs removed:", const_inputs)
        inputs = [c for c in inputs if c not in const_inputs]
    print(f"  inputs ({len(inputs)}): {inputs}")

    if RUN_EDA:
        explore(label, data, feats, cover, safe)

    # ---- save the model-ready table for step 2 ----
    keep_cols = list(dict.fromkeys([DEVICE_COL, "_t", "_variant"] + inputs + TARGETS + [f"{t}_future" for t in TARGETS]))
    out = data[keep_cols].copy()
    out[DEVICE_COL] = out[DEVICE_COL].astype("category")
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    path = PREPARED_DIR / f"prepared_{RUN_LABEL}_{safe}.pkl"
    out.to_pickle(path)
    sample_path = ""
    if SAVE_SAMPLE_CSV_ROWS:
        sample_path = str(PREPARED_DIR / f"prepared_sample_{RUN_LABEL}_{safe}.csv")
        out.head(SAVE_SAMPLE_CSV_ROWS).to_csv(sample_path, index=False)
    print(f"  saved {len(out)} rows x {len(keep_cols)} columns to {path}")
    return {"label": label, "safe": safe, "file": str(path), "sample_csv": sample_path, "inputs": inputs,
            "rows": int(len(out)), "vehicles": int(out[DEVICE_COL].nunique()),
            "variants": sorted(out["_variant"].astype(str).unique().tolist())}


# =====================================================================
# RUN: prepare all variants together and each group
# =====================================================================
jobs = []                                              # (label, devices of the group or None = everything, variant input)
n_groups = len(group_devices)
if RUN_CLASS_LEVEL or n_groups == 1:
    jobs.append((f"{CLASS_LABEL}_all", None, True))
if RUN_GROUP_LEVEL and n_groups > 1:
    for g_, devs in group_devices.items():
        jobs.append((f"{CLASS_LABEL}_{g_}", devs, True))     # a variant input is added only if the group holds 2+ variants

entries = []
for label, devs, variant_in in jobs:
    frame = sig if devs is None else sig[sig[DEVICE_COL].isin(devs)]
    entry = prepare_group(label, frame, variant_in)
    del frame
    if entry is not None:
        entries.append(entry)

if not entries:
    print("\nNo group could be prepared. Read the messages above (usually too little continuous data).")
    sys.exit(1)

index = {"run_label": RUN_LABEL, "device_col": DEVICE_COL, "targets": TARGETS, "grid_sec": GRID_SEC,
         "horizon_min": HORIZON_MIN, "class_label": CLASS_LABEL, "groups": entries}
index_path = PREPARED_DIR / f"prepared_index_{RUN_LABEL}.json"
with open(index_path, "w") as fh:
    json.dump(index, fh, indent=2)

banner("PRE-PROCESSING DONE")
print(pd.DataFrame([{k: e[k] for k in ("label", "rows", "vehicles", "variants")} for e in entries]).to_string(index=False))
print(f"\nPrepared tables and index: {PREPARED_DIR}")
print(f"Other results (files starting with out_) are in the folder of this script.")
print("Next step: training_validation_testing_mcu_motor_temp_pred.py")
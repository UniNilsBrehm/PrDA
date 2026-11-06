"""
pairwise_lagcorr_analysis.py
============================

Lag-resolved pairwise cross-correlation between neurons.

Outputs per-sweep and pooled figures:
  Figure 1 — raw correlation heatmaps + shuffle control panel
  Figure 2 — z-score heatmaps (real vs circular-shift null)

Each panel has two rows:
  Top    — heatmap (rows = neuron pairs, x = lag, colour = correlation / z-score)
  Bottom — mean ± 1 SD across all pairs plotted against lag

Shuffle control
---------------
Each neuron's trace is independently circularly shifted by a random amount
bounded away from zero (≥ MIN_SHIFT_FRACTION × window length), preserving
each neuron's autocorrelation structure while destroying cross-neuron timing.
N_SHUFFLES replicates are averaged to estimate the null mean and SD.

Optional features (toggle in PARAMETERS)
-----------------------------------------
SHUFFLE_ENTIRE_TRACE  — also run the circular-shift null on the entire-trace
                        condition (uses its own lag axis MAX_LAG_ENTIRE_TRACE_S)
                        so a zscore_entire_trace panel is produced.

SIGN_TEST_ZERO_LAG    — binomial sign test at zero lag: across all neuron pairs,
                        is C_real > C_null_mean more than 50 % of the time?
                        Result printed to console and saved in the summary CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binomtest

from config import Config
from utils import get_rois_per_sweep, load_hdf5_as_dict, get_rois_of_one_sweep


# ============================================================================ #
# PARAMETERS
# ============================================================================ #

# quiet_spont  = no stimulus nearby, no swim nearby
QUIET_SPONT_S = 20.0
QUIET_EXCLUSION_S = 5.0
# For each sweep, randomly select up to x quiet spontaneous windows
N_QUIET_WINDOWS_PER_SWEEP = 20

# Time before stimulus for pre-stimulus baseline condition
PRE_STIM_S = 20

# Cut out windows for stimulus onsets
PERI_PRE_STIM_S = 10.0
PERI_POST_STIM_S = 10.0

# Cut out windows for spont. swims
SPONT_PRE_S = 10.0
SPONT_POST_S = 10.0

# window length must be > MAX_LAG_S
MAX_LAG_S = 9
MAX_LAG_ENTIRE_TRACE_S = 30
MIN_FRAMES = 5
MIN_NEURONS = 3

MAX_PAIRS_PLOT = 5000
PAIR_SORT_MODE = "none"   # "peak_abs", "zero_lag", or "none"

# ---- Shuffle control & z-score ----------------------------------------------
N_SHUFFLES = 200
MIN_SHIFT_FRACTION = 0.2
ZSCORE_CLIM = 5.0

# ---- Optional features (set to False to disable) ----------------------------
SHUFFLE_ENTIRE_TRACE = True
SIGN_TEST_ZERO_LAG = True
SIGN_TEST_ALPHA = 0.05
# -----------------------------------------------------------------------------

EXCLUDE_FISH = []

OUT_DIR = f"{Config.BASE_DIR}/synchronization/pairwise_lagcorr"
FIG_DIR = f"{OUT_DIR}/figures"
CSV_DIR = f"{OUT_DIR}/tables"
POOLED_FIG        = f"{FIG_DIR}/ALL_FISH_pairwise_lagcorr_stacked.png"
POOLED_FIG_ZSCORE = f"{FIG_DIR}/ALL_FISH_pairwise_lagcorr_zscore.png"

REAL_CONDS = [
    "entire_trace",
    "quiet_spont",
    "spont_swim",
    "pre_stim",
    "visual_noswim",
    "visual_swim",
]

COND_LABELS_BASE = {
    "entire_trace":  "Entire trace",
    "quiet_spont":   "Quiet\nspontaneous",
    "spont_swim":    "Spontaneous\nswim",
    "pre_stim":      "Pre-stimulus",
    "visual_noswim": "Visual onset\n(no swim)",
    "visual_swim":   "Visual onset\n+ swim",
}

_zscore_source_conds = REAL_CONDS if SHUFFLE_ENTIRE_TRACE else [
    c for c in REAL_CONDS if c != "entire_trace"
]
ZSCORE_CONDS = [f"zscore_{c}" for c in _zscore_source_conds]

_entire_trace_shuffle_panel = ["shuffle_entire_trace"] if SHUFFLE_ENTIRE_TRACE else []
CORR_COND_ORDER = (
    ["entire_trace"]
    + _entire_trace_shuffle_panel
    + [c for c in REAL_CONDS if c != "entire_trace"]
    + ["shuffle_control"]
)

ZSCORE_COND_ORDER = ZSCORE_CONDS

COND_ORDER = CORR_COND_ORDER + ZSCORE_COND_ORDER

COND_LABELS = {
    **COND_LABELS_BASE,
    "shuffle_entire_trace": f"Shuffle control\nEntire trace\n(circ-shift, n={N_SHUFFLES})",
    "shuffle_control":      f"Shuffle control\n(circ-shift, n={N_SHUFFLES})",
    **{f"zscore_{c}": f"Z-score\n{COND_LABELS_BASE[c]}" for c in _zscore_source_conds},
}


# ============================================================================ #
# BASIC HELPERS
# ============================================================================ #

def get_sweep_to_fish(ca_labels, fish_row=0, sweep_row=1):
    sweeps = ca_labels.iloc[sweep_row].astype(str).values
    fish   = ca_labels.iloc[fish_row].astype(str).values
    mapping = {sw: f"fish{f}" for sw, f in zip(sweeps, fish)}

    print(f"[sweep->fish] {len(set(mapping.values()))} fish across {len(set(mapping))} sweeps")
    for sw in sorted(set(mapping)):
        print(f"    {sw} -> {mapping[sw]}")

    return mapping


def finite_nonflat(block):
    """Keep neurons with finite values and non-zero variance. Shape: neurons × time."""
    block = np.asarray(block, dtype=float)
    if block.ndim != 2:
        raise ValueError("block must be neurons × time")
    ok = np.all(np.isfinite(block), axis=1) & (np.nanstd(block, axis=1) > 0)
    return block[ok]


def zscore_rows(block):
    """Z-score each neuron across time."""
    block = finite_nonflat(block)
    if block.shape[0] == 0:
        return block
    mu = np.nanmean(block, axis=1, keepdims=True)
    sd = np.nanstd(block, axis=1, keepdims=True)
    sd[sd == 0] = np.nan
    Z = (block - mu) / sd
    Z = Z[np.all(np.isfinite(Z), axis=1)]
    return Z


def sort_limit_pairs_with_fish_ids(lags_s, C, fish_ids,
                                   max_pairs=MAX_PAIRS_PLOT, mode=PAIR_SORT_MODE):
    """Sort/limit pair rows and carry fish IDs along."""
    if C.size == 0:
        return C, fish_ids

    if mode == "peak_abs":
        score = np.nanmax(np.abs(C), axis=1)
        order = np.argsort(score)[::-1]
    elif mode == "zero_lag":
        zero_idx = int(np.argmin(np.abs(lags_s)))
        score = C[:, zero_idx]
        order = np.argsort(score)[::-1]
    elif mode == "none":
        order = np.arange(C.shape[0])
    else:
        raise ValueError(f"Unknown PAIR_SORT_MODE: {mode}")

    C_sorted    = C[order]
    fish_sorted = fish_ids[order]

    if max_pairs is not None and C_sorted.shape[0] > max_pairs:
        C_sorted    = C_sorted[:max_pairs]
        fish_sorted = fish_sorted[:max_pairs]

    return C_sorted, fish_sorted


def add_fish_identity_bar(ax, fish_ids):
    """Add a black/white side bar on the right indicating fish identity."""
    if len(fish_ids) == 0:
        return

    unique_fish  = list(dict.fromkeys(fish_ids))
    fish_to_value = {fish: i % 2 for i, fish in enumerate(unique_fish)}
    bar = np.array([fish_to_value[f] for f in fish_ids], dtype=float)[:, None]

    ax_bar = ax.inset_axes([1.015, 0.0, 0.025, 1.0])
    ax_bar.imshow(bar, aspect="auto", origin="lower", cmap="gray",
                  interpolation="nearest", vmin=0, vmax=1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_frame_on(False)


def stack_grouped_results_across_sweeps(all_grouped):
    pooled = {}

    for cond in COND_ORDER:
        is_zscore = cond.startswith("zscore_")
        Cs       = []
        fish_ids = []
        lags_ref = None
        total_windows = 0

        for item in all_grouped:
            fish    = item["fish"]
            grouped = item["grouped"]

            if cond not in grouped:
                continue

            entry  = grouped[cond]
            lags_s = entry["lags_s"]
            C      = entry["C_mean"]

            if C.size == 0:
                continue

            if lags_ref is None:
                lags_ref = lags_s
            elif len(lags_s) != len(lags_ref):
                continue

            Cs.append(C)
            if not is_zscore:
                fish_ids.extend([fish] * C.shape[0])
            total_windows += entry["n_windows"]

        if Cs:
            C_stacked = np.vstack(Cs)
            entry_out = dict(
                lags_s=lags_ref,
                C_mean=C_stacked,
                n_windows=total_windows,
                n_pairs=C_stacked.shape[0],
            )
            if not is_zscore:
                entry_out["fish_ids"] = np.asarray(fish_ids)

            if is_zscore and SIGN_TEST_ZERO_LAG:
                st = compute_sign_test_zero_lag(C_stacked, lags_ref)
                entry_out["sign_test_zero_lag"] = st

            pooled[cond] = entry_out

    return pooled


# ============================================================================ #
# WINDOW EXTRACTION
# ============================================================================ #

def extract_condition_windows(sw, fish, sw_ca_traces, motor_events,
                               sw_stimulus_onsets, fs):
    """Returns full-length condition windows only."""
    X = sw_ca_traces.values.T.astype(float)   # neurons × time

    if X.shape[0] < MIN_NEURONS:
        return []

    T = X.shape[1]

    pre_stim_frames  = int(round(PRE_STIM_S * fs))
    pre_frames       = int(round(PERI_PRE_STIM_S * fs))
    post_frames      = int(round(PERI_POST_STIM_S * fs))
    spont_pre_frames = int(round(SPONT_PRE_S * fs))
    spont_post_frames= int(round(SPONT_POST_S * fs))

    windows = []

    def add_window(cond, f0, f1, stim="none"):
        f0, f1 = int(f0), int(f1)
        if f0 < 0 or f1 > T or f1 - f0 < MIN_FRAMES:
            return
        block = X[:, f0:f1]
        windows.append(dict(
            fish=fish, sweep=sw, cond=cond, stim=stim,
            f0=f0, f1=f1,
            t0_s=float(f0 / fs), t1_s=float(f1 / fs),
            n_neurons=int(X.shape[0]), n_frames=int(f1 - f0),
            block=block,
        ))

    # Entire trace
    add_window("entire_trace", 0, T, stim="entire")

    # Visual stimulus windows
    for stim in sw_stimulus_onsets.columns:
        stim_on_s = sw_stimulus_onsets[stim].item()
        stim_f    = int(round(stim_on_s * fs))

        add_window("pre_stim", stim_f - pre_stim_frames, stim_f, stim=stim)

        f0 = stim_f - pre_frames
        f1 = stim_f + post_frames
        if f0 < 0 or f1 > T:
            continue

        if stim in motor_events.columns:
            swam = motor_events[stim].iloc[stim_f:f1].sum() > 0
        else:
            swam = False

        add_window("visual_swim" if swam else "visual_noswim", f0, f1, stim=stim)

    # Spontaneous swim windows
    if "spontaneous" in motor_events.columns:
        sp = np.where(motor_events["spontaneous"].values > 0)[0]
        if len(sp):
            min_separation = max(1, spont_post_frames // 2)
            onsets = sp[np.concatenate([[True], np.diff(sp) > min_separation])]
            for k, f_on in enumerate(onsets):
                add_window("spont_swim",
                           f_on - spont_pre_frames,
                           f_on + spont_post_frames,
                           stim=f"spont_{k:03d}")

    # Quiet spontaneous windows
    quiet_frames = int(round(QUIET_SPONT_S * fs))
    excl_frames  = int(round(QUIET_EXCLUSION_S * fs))
    bad = np.zeros(T, dtype=bool)

    for stim in sw_stimulus_onsets.columns:
        stim_f = int(round(sw_stimulus_onsets[stim].item() * fs))
        bad[max(0, stim_f - excl_frames):min(T, stim_f + excl_frames)] = True

    for col in motor_events.columns:
        for f in np.where(motor_events[col].values > 0)[0]:
            bad[max(0, f - excl_frames):min(T, f + excl_frames)] = True

    valid_starts = [f0 for f0 in range(0, T - quiet_frames)
                    if not bad[f0:f0 + quiet_frames].any()]

    if valid_starts:
        rng    = np.random.default_rng(123)
        chosen = rng.choice(valid_starts,
                            size=min(N_QUIET_WINDOWS_PER_SWEEP, len(valid_starts)),
                            replace=False)
        for k, f0 in enumerate(sorted(chosen)):
            add_window("quiet_spont", f0, f0 + quiet_frames, stim=f"quiet_{k:03d}")

    return windows


# ============================================================================ #
# PAIRWISE LAG CORRELATION
# ============================================================================ #

def compute_pairwise_lagcorr(block, fs, max_lag_s=MAX_LAG_S):
    """Faster pairwise lag correlation via lagged dot-product on z-scored traces."""
    Z = zscore_rows(block)
    n, T = Z.shape

    if n < 2 or T < MIN_FRAMES:
        return None, [], np.empty((0, 0))

    requested_max_lag = int(round(max_lag_s * fs))
    if T - requested_max_lag < MIN_FRAMES:
        return None, [], np.empty((0, 0))

    lag_frames = np.arange(-requested_max_lag, requested_max_lag + 1)
    lags_s     = lag_frames / fs

    triu  = np.triu_indices(n, k=1)
    pairs = list(zip(triu[0], triu[1]))
    C     = np.empty((len(pairs), len(lag_frames)), dtype=float)

    for li, lag in enumerate(lag_frames):
        if lag < 0:
            A, B = Z[:, :lag], Z[:, -lag:]
        elif lag > 0:
            A, B = Z[:, lag:], Z[:, :-lag]
        else:
            A, B = Z, Z
        M = (A @ B.T) / A.shape[1]
        C[:, li] = M[triu]

    return lags_s, pairs, C


# ============================================================================ #
# SHUFFLE CONTROL
# ============================================================================ #

def circshift_block(block, rng, min_shift_fraction=MIN_SHIFT_FRACTION):
    """Independently circularly shift each neuron by a random amount."""
    n, T      = block.shape
    min_shift = max(1, int(T * min_shift_fraction))
    max_shift = T - min_shift

    if max_shift <= min_shift:
        shifts = rng.integers(low=1, high=T, size=n)
    else:
        shifts = rng.integers(low=min_shift, high=max_shift, size=n)

    return np.stack([np.roll(block[i], shifts[i]) for i in range(n)], axis=0)


def compute_shuffle_lagcorr(block, fs, max_lag_s=MAX_LAG_S,
                             n_shuffles=N_SHUFFLES, seed=0):
    """
    Circular-shift null: returns mean and SD of cross-correlation across replicates.
    """
    rng = np.random.default_rng(seed)
    lags_s, pairs, C_first = compute_pairwise_lagcorr(block, fs, max_lag_s=max_lag_s)

    if C_first.size == 0:
        empty = np.empty((0, 0))
        return lags_s, pairs, empty, empty

    C_accum    = np.zeros_like(C_first)
    C_accum_sq = np.zeros_like(C_first)
    n_valid    = 0

    for _ in range(n_shuffles):
        shuffled = circshift_block(block, rng)
        _, _, C_shuf = compute_pairwise_lagcorr(shuffled, fs, max_lag_s=max_lag_s)
        if C_shuf.size == 0 or C_shuf.shape != C_first.shape:
            continue
        C_accum    += C_shuf
        C_accum_sq += C_shuf ** 2
        n_valid    += 1

    if n_valid == 0:
        empty = np.empty((0, 0))
        return lags_s, pairs, empty, empty

    C_null_mean = C_accum / n_valid
    C_null_std  = np.sqrt(np.maximum(C_accum_sq / n_valid - C_null_mean ** 2, 0.0))

    return lags_s, pairs, C_null_mean, C_null_std


# ============================================================================ #
# SIGN TEST (zero-lag only)
# ============================================================================ #

def compute_sign_test_zero_lag(C_real, lags_s, C_null_mean=None,
                                alpha=SIGN_TEST_ALPHA):
    """
    Two-sided binomial sign test at zero lag.

    If C_null_mean is provided, tests sign(C_real - C_null_mean) at zero lag.
    Otherwise tests sign(C_real) > 0 (i.e. against zero correlation).
    """
    zero_idx = int(np.argmin(np.abs(lags_s)))
    real_zero = C_real[:, zero_idx]

    if C_null_mean is not None:
        null_zero = C_null_mean[:, zero_idx]
        n_pos = int(np.sum(real_zero > null_zero))
    else:
        n_pos = int(np.sum(real_zero > 0))

    n_pairs = len(real_zero)
    p = binomtest(n_pos, n_pairs, p=0.5, alternative="two-sided").pvalue

    return dict(p=p, n_pos=n_pos, n_pairs=n_pairs)


def summarize_lagcorr(lags_s, C):
    """Compact summary statistics for one condition/window."""
    if C.size == 0:
        return dict(mean_zero_lag=np.nan, mean_peak_abs=np.nan,
                    mean_peak_lag_s=np.nan, n_pairs=0)

    zero_idx  = int(np.argmin(np.abs(lags_s)))
    zero_vals = C[:, zero_idx]
    abs_C     = np.abs(C)
    peak_idx  = np.nanargmax(abs_C, axis=1)
    peak_vals = C[np.arange(C.shape[0]), peak_idx]
    peak_lags = lags_s[peak_idx]

    return dict(
        mean_zero_lag=float(np.nanmean(zero_vals)),
        mean_peak_abs=float(np.nanmean(np.abs(peak_vals))),
        mean_peak_lag_s=float(np.nanmean(peak_lags)),
        n_pairs=int(C.shape[0]),
    )


# ============================================================================ #
# PER-SWEEP ANALYSIS
# ============================================================================ #

def average_condition_lagcorr(windows, fs):
    """
    Average lag-correlation matrices across windows of the same condition,
    then compute shuffle null, z-scores, and optional sign test.
    """
    grouped = {}

    # ---- Real conditions ---------------------------------------------------- #
    for cond in REAL_CONDS:
        cond_windows = [w for w in windows if w["cond"] == cond]
        if not cond_windows:
            continue

        Cs        = []
        lags_ref  = None
        pairs_ref = None

        for w in cond_windows:
            max_lag_s = MAX_LAG_ENTIRE_TRACE_S if cond == "entire_trace" else MAX_LAG_S
            lags_s, pairs, C = compute_pairwise_lagcorr(w["block"], fs,
                                                         max_lag_s=max_lag_s)
            if C.size == 0:
                continue
            if lags_ref is None:
                lags_ref, pairs_ref = lags_s, pairs
            elif len(lags_s) != len(lags_ref):
                continue
            Cs.append(C)

        if not Cs:
            continue

        C_mean = np.nanmean(np.stack(Cs, axis=0), axis=0)
        grouped[cond] = dict(lags_s=lags_ref, pairs=pairs_ref,
                             C_mean=C_mean, n_windows=len(Cs),
                             n_pairs=C_mean.shape[0])

    # ---- Shuffle control & z-scores ----------------------------------------- #
    def _run_shuffle_for_conds(source_conds, max_lag_s, seed_start=0):
        source_windows = [w for w in windows if w["cond"] in source_conds]
        if not source_windows:
            return seed_start

        Cs_null    = []
        Cs_null_sq = []
        lags_ref   = None
        pairs_ref  = None
        seed_offset = seed_start

        for w in source_windows:
            lags_s, pairs, C_null_mean_w, C_null_std_w = compute_shuffle_lagcorr(
                w["block"], fs, max_lag_s=max_lag_s,
                n_shuffles=N_SHUFFLES, seed=seed_offset,
            )
            seed_offset += 1
            if C_null_mean_w.size == 0:
                continue
            if lags_ref is None:
                lags_ref, pairs_ref = lags_s, pairs
            elif len(lags_s) != len(lags_ref):
                continue
            Cs_null.append(C_null_mean_w)
            Cs_null_sq.append(C_null_std_w ** 2)

        if not Cs_null:
            return seed_offset

        C_null_mean = np.nanmean(np.stack(Cs_null, axis=0), axis=0)
        C_null_std  = np.sqrt(np.nanmean(np.stack(Cs_null_sq, axis=0), axis=0))

        if max_lag_s == MAX_LAG_S:
            grouped["shuffle_control"] = dict(
                lags_s=lags_ref, pairs=pairs_ref,
                C_mean=C_null_mean, C_std=C_null_std,
                n_windows=len(Cs_null), n_pairs=C_null_mean.shape[0],
            )
        else:
            grouped["shuffle_entire_trace"] = dict(
                lags_s=lags_ref, pairs=pairs_ref,
                C_mean=C_null_mean, C_std=C_null_std,
                n_windows=len(Cs_null), n_pairs=C_null_mean.shape[0],
            )

        for cond in source_conds:
            if cond not in grouped:
                continue
            C_real = grouped[cond]["C_mean"]
            if C_real.shape != C_null_mean.shape:
                continue

            with np.errstate(invalid="ignore", divide="ignore"):
                C_z = np.where(C_null_std > 0,
                               (C_real - C_null_mean) / C_null_std,
                               0.0)

            zscore_entry = dict(
                lags_s=lags_ref, pairs=pairs_ref,
                C_mean=C_z, C_null_mean=C_null_mean,
                n_windows=grouped[cond]["n_windows"],
                n_pairs=C_z.shape[0],
            )

            if SIGN_TEST_ZERO_LAG:
                st = compute_sign_test_zero_lag(C_real, lags_ref, C_null_mean)
                zscore_entry["sign_test_zero_lag"] = st
                print(f"  [sign test zero-lag] {cond}: "
                      f"p={st['p']:.4g}, "
                      f"{st['n_pos']}/{st['n_pairs']} pairs positive")

            grouped[f"zscore_{cond}"] = zscore_entry

        return seed_offset

    non_entire = [c for c in REAL_CONDS if c != "entire_trace"]
    next_seed  = _run_shuffle_for_conds(non_entire, max_lag_s=MAX_LAG_S, seed_start=0)

    if SHUFFLE_ENTIRE_TRACE and "entire_trace" in grouped:
        _run_shuffle_for_conds(["entire_trace"],
                               max_lag_s=MAX_LAG_ENTIRE_TRACE_S,
                               seed_start=next_seed)

    return grouped


def process_one_sweep(sw, fish, sw_ca_traces, motor_events,
                      sw_stimulus_onsets, fs):
    windows = extract_condition_windows(
        sw=sw, fish=fish, sw_ca_traces=sw_ca_traces,
        motor_events=motor_events,
        sw_stimulus_onsets=sw_stimulus_onsets, fs=fs,
    )
    grouped = average_condition_lagcorr(windows, fs)

    summary_rows = []
    for cond, entry in grouped.items():
        s   = summarize_lagcorr(entry["lags_s"], entry["C_mean"])
        row = dict(fish=fish, sweep=sw, cond=cond,
                   n_windows=entry["n_windows"], **s)
        if SIGN_TEST_ZERO_LAG and "sign_test_zero_lag" in entry:
            st = entry["sign_test_zero_lag"]
            row["sign_test_zero_lag_p"]       = st["p"]
            row["sign_test_zero_lag_n_pos"]   = st["n_pos"]
            row["sign_test_zero_lag_n_pairs"]  = st["n_pairs"]
        summary_rows.append(row)

    return grouped, pd.DataFrame(summary_rows)


# ============================================================================ #
# PLOTTING HELPERS
# ============================================================================ #

def sort_and_limit_pairs(lags_s, C, max_pairs=MAX_PAIRS_PLOT, mode=PAIR_SORT_MODE):
    """Sort pair rows for visualization and optionally limit number of pairs."""
    if C.size == 0:
        return C

    if mode == "peak_abs":
        score = np.nanmax(np.abs(C), axis=1)
        order = np.argsort(score)[::-1]
    elif mode == "zero_lag":
        zero_idx = int(np.argmin(np.abs(lags_s)))
        score    = C[:, zero_idx]
        order    = np.argsort(score)[::-1]
    elif mode == "none":
        order = np.arange(C.shape[0])
    else:
        raise ValueError(f"Unknown PAIR_SORT_MODE: {mode}")

    C_sorted = C[order]
    if max_pairs is not None and C_sorted.shape[0] > max_pairs:
        C_sorted = C_sorted[:max_pairs]
    return C_sorted


def _plot_heatmap(ax, lags_s, C, title, zscore_mode=False):
    """Draw the heatmap into ax. Returns the imshow object."""
    if C.size == 0:
        ax.text(0.5, 0.5, "No valid pairs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Neuron pair")
        return None

    C_plot = sort_and_limit_pairs(lags_s, C)

    if zscore_mode:
        cmap, vmin, vmax = "RdBu_r", -ZSCORE_CLIM, ZSCORE_CLIM
    else:
        cmap, vmin, vmax = "coolwarm", -1, 1

    im = ax.imshow(C_plot, aspect="auto", origin="lower",
                   extent=[lags_s[0], lags_s[-1], 0, C_plot.shape[0]],
                   vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Neuron pair")
    ax.tick_params(labelbottom=False)
    return im


def _plot_mean_trace(ax, lags_s, C, zscore_mode=False, first_col=True, ylim=None):
    if C.size == 0:
        ax.set_xlabel("Lag (s)")
        return

    mean = np.nanmean(C, axis=0)
    sd   = np.nanstd(C, axis=0)

    color = "#c0392b" if zscore_mode else "#2c7bb6"

    ax.fill_between(lags_s, mean - sd, mean + sd, alpha=0.25, color=color)
    ax.plot(lags_s, mean, color=color, lw=1.2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Lag (s)")
    ax.set_xlim(lags_s[0], lags_s[-1])

    if ylim is not None:
        ax.set_ylim(ylim)

    ylabel = "Mean Z-score" if zscore_mode else "Mean corr."
    if first_col:
        ax.set_ylabel(ylabel, fontsize=8)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="both", labelsize=7)


def _shared_ylim(entries, cond_list, pad=0.05):
    all_lo, all_hi = [], []
    for cond in cond_list:
        entry = entries.get(cond)
        if entry is None or entry["C_mean"].size == 0:
            continue
        C    = entry["C_mean"]
        mean = np.nanmean(C, axis=0)
        sd   = np.nanstd(C, axis=0)
        all_lo.append(np.nanmin(mean - sd))
        all_hi.append(np.nanmax(mean + sd))

    if not all_lo:
        return None

    lo, hi = min(all_lo), max(all_hi)
    span   = hi - lo if hi != lo else 1.0
    return (lo - pad * span, hi + pad * span)


def _make_panel_grid(fig, n_panels, heatmap_fraction=0.78):
    outer = gridspec.GridSpec(1, n_panels, figure=fig,
                              left=0.06, right=0.88,
                              bottom=0.10, top=0.78,
                              wspace=0.35)
    panels = []
    for i in range(n_panels):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[i],
            height_ratios=[heatmap_fraction, 1 - heatmap_fraction],
            hspace=0.08,
        )
        ax_heat = fig.add_subplot(inner[0])
        ax_mean = fig.add_subplot(inner[1], sharex=ax_heat)
        panels.append((ax_heat, ax_mean))
    return panels


# ============================================================================ #
# FIGURE 1 — raw correlation + shuffle control
# ============================================================================ #

def plot_sweep_lagcorr_figure(sw, fish, grouped_results, save_path=None):
    """Per-sweep: raw correlation heatmaps + shuffle control. coolwarm ±1."""
    cond_list = CORR_COND_ORDER
    n_panels  = len(cond_list)

    fig    = plt.figure(figsize=(4.1 * n_panels, 6.0))
    panels = _make_panel_grid(fig, n_panels)
    last_im = None

    ylim = _shared_ylim(grouped_results, cond_list)

    for pi, ((ax_h, ax_m), cond) in enumerate(zip(panels, cond_list)):
        entry = grouped_results.get(cond)

        if entry is None:
            ax_h.text(0.5, 0.5, "No windows", ha="center", va="center",
                      transform=ax_h.transAxes)
            ax_h.set_title(COND_LABELS[cond], fontsize=10, fontweight="bold")
            ax_m.set_xlabel("Lag (s)")
            continue

        lags_s = entry["lags_s"]
        C      = entry["C_mean"]
        title  = (f"{COND_LABELS[cond]}\n"
                  f"{entry['n_windows']} window(s), {entry['n_pairs']} pairs")

        im = _plot_heatmap(ax_h, lags_s, C, title, zscore_mode=False)
        if im is not None:
            last_im = im

        _plot_mean_trace(ax_m, lags_s, C, zscore_mode=False,
                         first_col=(pi == 0), ylim=ylim)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.22, 0.012, 0.52])
        fig.colorbar(last_im, cax=cbar_ax).set_label("Cross-correlation")

    fig.suptitle(f"Pairwise lag cross-correlation — {sw} / {fish}",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02,
             f"Top: heatmap (rows = pairs). "
             f"Bottom: mean ± 1 SD across pairs. "
             f"Shuffle: {N_SHUFFLES} circ-shift replicates.",
             ha="center", fontsize=8, color="0.35")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved figure -> {save_path}")

    return fig


def plot_pooled_lagcorr_figure(pooled_results, save_path=None):
    """Pooled across all fish: raw correlation heatmaps + shuffle control."""
    cond_list = CORR_COND_ORDER
    n_panels  = len(cond_list)

    fig    = plt.figure(figsize=(4.1 * n_panels, 6.4))
    panels = _make_panel_grid(fig, n_panels)
    last_im = None

    ylim = _shared_ylim(pooled_results, cond_list)

    for pi, ((ax_h, ax_m), cond) in enumerate(zip(panels, cond_list)):
        entry = pooled_results.get(cond)

        if entry is None:
            ax_h.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_h.transAxes)
            ax_h.set_title(COND_LABELS[cond], fontsize=10, fontweight="bold")
            ax_m.set_xlabel("Lag (s)")
            continue

        lags_s   = entry["lags_s"]
        C        = entry["C_mean"]
        fish_ids = entry.get("fish_ids", np.array([]))

        if len(fish_ids) > 0:
            C_plot, fish_plot = sort_limit_pairs_with_fish_ids(lags_s, C, fish_ids)
        else:
            C_plot    = sort_and_limit_pairs(lags_s, C)
            fish_plot = np.array([])

        title = (f"{COND_LABELS[cond]}\n{entry['n_windows']} window(s), "
                 f"{entry['n_pairs']} pairs")

        im = _plot_heatmap(ax_h, lags_s, C_plot, title, zscore_mode=False)
        if im is not None:
            last_im = im

        if len(fish_plot) > 0:
            add_fish_identity_bar(ax_h, fish_plot)

        _plot_mean_trace(ax_m, lags_s, C, zscore_mode=False,
                         first_col=(pi == 0), ylim=ylim)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.22, 0.012, 0.52])
        fig.colorbar(last_im, cax=cbar_ax).set_label("Cross-correlation")

    fig.suptitle("Pairwise lag cross-correlation — all fish stacked",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02,
             f"Top: heatmap (black/white bar = fish identity). "
             f"Bottom: mean ± 1 SD across pairs (shared y-axis). "
             f"Pair sorting: {PAIR_SORT_MODE}. "
             f"Shuffle: {N_SHUFFLES} circ-shift replicates.",
             ha="center", fontsize=8, color="0.35")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved pooled figure -> {save_path}")

    return fig


# ============================================================================ #
# FIGURE 2 — z-score heatmaps
# ============================================================================ #

def plot_sweep_zscore_figure(sw, fish, grouped_results, save_path=None):
    """Per-sweep: z-score heatmaps. RdBu_r ±ZSCORE_CLIM."""
    cond_list = ZSCORE_COND_ORDER
    n_panels  = len(cond_list)

    if n_panels == 0:
        return None

    fig    = plt.figure(figsize=(4.1 * n_panels, 6.0))
    panels = _make_panel_grid(fig, n_panels)
    last_im = None

    ylim = _shared_ylim(grouped_results, cond_list)

    for pi, ((ax_h, ax_m), cond) in enumerate(zip(panels, cond_list)):
        entry = grouped_results.get(cond)

        if entry is None:
            ax_h.text(0.5, 0.5, "No windows", ha="center", va="center",
                      transform=ax_h.transAxes)
            ax_h.set_title(COND_LABELS[cond], fontsize=10, fontweight="bold")
            ax_m.set_xlabel("Lag (s)")
            continue

        lags_s = entry["lags_s"]
        C      = entry["C_mean"]
        title  = (f"{COND_LABELS[cond]}\n"
                  f"{entry['n_windows']} window(s), {entry['n_pairs']} pairs")

        im = _plot_heatmap(ax_h, lags_s, C, title, zscore_mode=True)
        if im is not None:
            last_im = im

        _plot_mean_trace(ax_m, lags_s, C, zscore_mode=True,
                         first_col=(pi == 0), ylim=ylim)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.22, 0.012, 0.52])
        fig.colorbar(last_im, cax=cbar_ax).set_label("Z-score (shuffle SD)")

    fig.suptitle(f"Pairwise lag cross-correlation Z-score — {sw} / {fish}",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02,
             f"Top: Z-score heatmap. Bottom: mean ± 1 SD across pairs (shared y-axis). "
             f"Z = (real − null mean) / null SD. "
             f"Null: {N_SHUFFLES} circ-shift replicates. "
             f"Color clipped at ±{ZSCORE_CLIM}.",
             ha="center", fontsize=8, color="0.35")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved figure -> {save_path}")

    return fig


def plot_pooled_zscore_figure(pooled_results, save_path=None):
    """Pooled across all fish: z-score heatmaps."""
    cond_list = ZSCORE_COND_ORDER
    n_panels  = len(cond_list)

    if n_panels == 0:
        return None

    fig    = plt.figure(figsize=(4.1 * n_panels, 6.4))
    panels = _make_panel_grid(fig, n_panels)
    last_im = None

    ylim = _shared_ylim(pooled_results, cond_list)

    for pi, ((ax_h, ax_m), cond) in enumerate(zip(panels, cond_list)):
        entry = pooled_results.get(cond)

        if entry is None:
            ax_h.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_h.transAxes)
            ax_h.set_title(COND_LABELS[cond], fontsize=10, fontweight="bold")
            ax_m.set_xlabel("Lag (s)")
            continue

        lags_s = entry["lags_s"]
        C      = entry["C_mean"]
        C_plot = sort_and_limit_pairs(lags_s, C)
        title  = (f"{COND_LABELS[cond]}\n{entry['n_windows']} window(s), "
                  f"{entry['n_pairs']} pairs")

        im = _plot_heatmap(ax_h, lags_s, C_plot, title, zscore_mode=True)
        if im is not None:
            last_im = im

        _plot_mean_trace(ax_m, lags_s, C, zscore_mode=True,
                         first_col=(pi == 0), ylim=ylim)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.22, 0.012, 0.52])
        fig.colorbar(last_im, cax=cbar_ax).set_label("Z-score (shuffle SD)")

    fig.suptitle("Pairwise lag cross-correlation Z-score — all fish stacked",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02,
             f"Top: Z-score heatmap. Bottom: mean ± 1 SD across pairs (shared y-axis). "
             f"Z = (real − null mean) / null SD. "
             f"Null: {N_SHUFFLES} circ-shift replicates. "
             f"Color clipped at ±{ZSCORE_CLIM}. "
             f"Pair sorting: {PAIR_SORT_MODE}.",
             ha="center", fontsize=8, color="0.35")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved pooled z-score figure -> {save_path}")

    return fig


# ============================================================================ #
# SERIALISATION HELPERS
# ============================================================================ #

def _matrix_path(sw, cond):
    """Path for the C_mean matrix CSV for one sweep × condition."""
    return f"{CSV_DIR}/{sw}_{cond}_C_mean.csv"


def _meta_path():
    """Path for the per-sweep condition metadata table."""
    return f"{CSV_DIR}/pairwise_lagcorr_metadata.csv"


def _pooled_path(cond):
    """Path for the pooled (all-sweeps) CSV for one condition."""
    return f"{CSV_DIR}/pooled_{cond}.csv"


def save_grouped(sw, fish, grouped):
    """
    Persist one sweep's grouped results to disk.

    For each condition:
      - C_mean matrix  → CSV  (rows = pairs, columns = lag values in seconds)
      - Metadata row   → appended to the shared metadata table
    """
    rows = []
    for cond, entry in grouped.items():
        C      = entry["C_mean"]
        lags_s = entry["lags_s"]

        mat_df = pd.DataFrame(C, columns=[f"{l:.6f}" for l in lags_s])
        mat_df.to_csv(_matrix_path(sw, cond), index=False)

        rows.append(dict(
            sweep     = sw,
            fish      = fish,
            cond      = cond,
            n_windows = entry["n_windows"],
            n_pairs   = entry["n_pairs"],
            n_lags    = len(lags_s),
        ))

    return pd.DataFrame(rows)


def load_grouped_for_sweep(sw, fish, meta_sw):
    """
    Reconstruct the `grouped` dict for one sweep from saved CSVs.
    """
    grouped = {}
    for _, row in meta_sw.iterrows():
        cond = row["cond"]
        path = _matrix_path(sw, cond)

        if not os.path.exists(path):
            print(f"  [load] missing matrix file: {path} — skipping {cond}")
            continue

        mat_df = pd.read_csv(path)
        lags_s = mat_df.columns.astype(float).values
        C      = mat_df.values

        grouped[cond] = dict(
            lags_s    = lags_s,
            C_mean    = C,
            n_windows = int(row["n_windows"]),
            n_pairs   = int(row["n_pairs"]),
        )

    return grouped


def load_pooled_condition(cond):
    """
    Load a pooled condition CSV back into arrays ready for plotting.

    Returns
    -------
    lags_s   : np.ndarray, shape (n_lags,)
    C        : np.ndarray, shape (n_pairs_total, n_lags)
    fish_ids : np.ndarray of str, shape (n_pairs_total,)
    sweeps   : np.ndarray of str, shape (n_pairs_total,)
    """
    path = _pooled_path(cond)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pooled CSV not found: {path}\n"
            f"Run main_pool() first."
        )

    df       = pd.read_csv(path)
    fish_ids = df["fish"].values.astype(str)
    sweeps   = df["sweep"].values.astype(str)
    data_df  = df.drop(columns=["fish", "sweep", "pair_idx"])
    lags_s   = data_df.columns.astype(float).values
    C        = data_df.values.astype(float)

    return lags_s, C, fish_ids, sweeps


# ============================================================================ #
# MAIN — COMPUTE
# ============================================================================ #

def main_compute():
    """
    Run all computations and save results to CSV files.

    Outputs
    -------
    {CSV_DIR}/{sw}_{cond}_C_mean.csv      — one per sweep × condition
    {CSV_DIR}/pairwise_lagcorr_metadata.csv
    {CSV_DIR}/pairwise_lagcorr_summary.csv
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    ca_df_f          = pd.read_csv(Config.ca_df_f_file)
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)["sampling_rate"].item()
    ca_labels        = pd.read_csv(Config.ca_labels_file)

    rois_per_sweep = get_rois_per_sweep(ca_labels)
    vr_binaries    = load_hdf5_as_dict(Config.vr_all_binaries_file)
    sweep_to_fish  = get_sweep_to_fish(ca_labels)

    all_summary  = []
    all_meta     = []

    for sw in rois_per_sweep:
        fish = sweep_to_fish[sw]

        if fish in EXCLUDE_FISH:
            print(f"skipping excluded fish: {fish} / {sw}")
            continue

        print(f"\nprocessing {sw} / {fish}")

        sw_rois      = get_rois_of_one_sweep(sw, rois_per_sweep)
        sw_ca_traces = ca_df_f[sw_rois]
        motor_events = vr_binaries[sw]

        sw_stimulus_onsets = pd.read_csv(
            f"{Config.BASE_DIR}/stimulus_traces/{sw}_stimulus_onsets.csv",
            index_col=0,
        )

        grouped, summary = process_one_sweep(
            sw=sw, fish=fish,
            sw_ca_traces=sw_ca_traces,
            motor_events=motor_events,
            sw_stimulus_onsets=sw_stimulus_onsets,
            fs=ca_sampling_rate,
        )

        meta = save_grouped(sw, fish, grouped)
        all_meta.append(meta)

        if len(summary):
            all_summary.append(summary)

    meta_df = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    meta_df.to_csv(_meta_path(), index=False)
    print(f"\nsaved metadata -> {_meta_path()}")

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    summary_path = f"{CSV_DIR}/pairwise_lagcorr_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"saved summary  -> {summary_path}")

    return meta_df, summary_df


# ============================================================================ #
# MAIN — POOL
# ============================================================================ #

def main_pool(cond_filter=None):
    """
    Read existing per-sweep C_mean CSVs and write one pooled CSV per condition.
    No recomputation — requires main_compute() to have been run first.

    Output columns in each pooled CSV:
        fish        — fish identifier
        sweep       — sweep identifier
        pair_idx    — within-sweep pair index (0-based)
        <lag_0>     — correlation / z-score value at lag = lag_0 seconds
        <lag_1>, ...

    Saved to:
        {CSV_DIR}/pooled_{cond}.csv   — one file per condition

    Parameters
    ----------
    cond_filter : list of str or None
        If given, only pool the listed conditions.
        If None, pool every condition found in the metadata index.
    """
    os.makedirs(CSV_DIR, exist_ok=True)

    meta_path = _meta_path()
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata index not found: {meta_path}\n"
            f"Run main_compute() first."
        )

    meta_df = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(meta_df)} rows, "
          f"{meta_df['sweep'].nunique()} sweeps, "
          f"{meta_df['cond'].nunique()} conditions.")

    conditions = meta_df["cond"].unique()
    if cond_filter is not None:
        conditions = [c for c in conditions if c in cond_filter]

    saved = {}

    for cond in sorted(conditions):
        meta_cond   = meta_df[meta_df["cond"] == cond]
        sweep_frames = []

        print(f"\npooling condition: {cond}  ({len(meta_cond)} sweeps in metadata)")

        for _, row in meta_cond.iterrows():
            sw   = str(row["sweep"])
            fish = str(row["fish"])
            path = _matrix_path(sw, cond)

            if not os.path.exists(path):
                print(f"  [pool] missing file: {path} — skipping sweep '{sw}'")
                continue

            mat_df  = pd.read_csv(path)               # rows=pairs, cols=lags
            n_pairs = len(mat_df)

            mat_df.insert(0, "pair_idx", np.arange(n_pairs, dtype=int))
            mat_df.insert(0, "sweep",    sw)
            mat_df.insert(0, "fish",     fish)

            sweep_frames.append(mat_df)

        if not sweep_frames:
            print(f"  [pool] no data found for condition '{cond}' — skipping")
            continue

        pooled_df = pd.concat(sweep_frames, ignore_index=True)
        out_path  = _pooled_path(cond)
        pooled_df.to_csv(out_path, index=False)

        n_pairs = len(pooled_df)
        n_lags  = len(pooled_df.columns) - 3          # subtract fish, sweep, pair_idx
        print(f"  saved -> {out_path}  ({n_pairs} pairs × {n_lags} lags)")
        saved[cond] = out_path

    print(f"\ndone. {len(saved)} pooled CSVs written.")
    return saved


# ============================================================================ #
# MAIN — PLOT
# ============================================================================ #

def main_plot():
    """
    Load saved CSV results and regenerate all figures — no recomputation.
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    meta_path = _meta_path()
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}\n"
            f"Run main_compute() first."
        )

    meta_df = pd.read_csv(meta_path)

    all_grouped = []

    for sw, meta_sw in meta_df.groupby("sweep"):
        fish = meta_sw["fish"].iloc[0]

        print(f"loading {sw} / {fish}")
        grouped = load_grouped_for_sweep(sw, fish, meta_sw)

        if not grouped:
            print(f"  no data found for {sw} — skipping figures")
            continue

        all_grouped.append(dict(fish=fish, sweep=sw, grouped=grouped))

        plot_sweep_lagcorr_figure(
            sw=sw, fish=fish, grouped_results=grouped,
            save_path=f"{FIG_DIR}/{sw}_pairwise_lagcorr.png",
        )
        plot_sweep_zscore_figure(
            sw=sw, fish=fish, grouped_results=grouped,
            save_path=f"{FIG_DIR}/{sw}_pairwise_lagcorr_zscore.png",
        )
        plt.close("all")

    if all_grouped:
        pooled = stack_grouped_results_across_sweeps(all_grouped)
        plot_pooled_lagcorr_figure(pooled_results=pooled, save_path=POOLED_FIG)
        plot_pooled_zscore_figure(pooled_results=pooled, save_path=POOLED_FIG_ZSCORE)
        plt.close("all")

    print("\ndone.")


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pairwise lag cross-correlation analysis."
    )
    parser.add_argument(
        "mode",
        choices=["compute", "pool", "plot", "both"],
        nargs="?",
        default="both",
        help=(
            "compute — run computations and save per-sweep CSVs; "
            "pool    — read per-sweep CSVs and write one pooled CSV per condition; "
            "plot    — load CSVs and regenerate figures only; "
            "both    — compute, pool, then plot (default)"
        ),
    )
    args = parser.parse_args()

    if args.mode in ("compute", "both"):
        main_compute()

    if args.mode in ("pool", "both"):
        main_pool()

    if args.mode in ("plot", "both"):
        main_plot()

"""
zscore_population_activity.py
================================

Compute event-aligned, z-score-thresholded population activity measures:

1. Fraction of active neurons
       k(t) / N

2. Fraction of jointly active neuron pairs (raw co-activity)
       k(t) * (k(t) - 1) / [N * (N - 1)]

The script keeps the original condition definitions:
    visual_noswim  - visual stimulus without a swim response
    visual_swim    - visual stimulus followed by swimming
    spont_swim     - spontaneous swim without a visual stimulus

Outputs
-------
- One CSV row per frame per event window.
- One fish-averaged time-course figure for fraction active.
- One fish-averaged time-course figure for raw co-activity.

Expected project modules
------------------------
- config.Config
- utils.get_rois_per_sweep
- utils.load_hdf5_as_dict
- utils.get_rois_of_one_sweep
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from utils import get_rois_per_sweep, get_rois_of_one_sweep, load_hdf5_as_dict


# ============================================================================
# Parameters
# ============================================================================
PRE_S = 5.0
POST_S = 15.0
REQUIRE_FULL_EVENT_WINDOW = True
MIN_FRAMES = 5
MIN_NEURONS = 3
Z_THRESHOLD = 0.5

OUT_DIR = f"{Config.BASE_DIR}/zscore_population_activity"
TRACE_CSV = f"{OUT_DIR}/zscore_population_activity_timecourses.csv"
FRACTION_ACTIVE_FIG = f"{OUT_DIR}/fraction_active_fish_average.png"
COACTIVITY_FIG = f"{OUT_DIR}/coactivity_fish_average.png"

COND_ORDER = ["visual_noswim", "spont_swim", "visual_swim"]
COND_LABELS = {
    "visual_noswim": "Visual (no swim)",
    "spont_swim": "Spontaneous swim",
    "visual_swim": "Visual + swim",
}
COND_COLORS = {
    "visual_noswim": "#4C72B0",
    "spont_swim": "#55A868",
    "visual_swim": "#C44E52",
}

# Optional manual event-alignment corrections in seconds.
EVENT_TIME_SHIFT_S = {
    "fish1": -2.2,
    "fish3": -0.2,
    "fish4": -0.2,
    "fish5": -0.2,
    "fish6": -0.2,
    "fish9": -0.2,
}

# Set to [] to include every fish.
EXCLUDE_FISH: list[str] = []


# ============================================================================
# Helpers
# ============================================================================
def get_sweep_to_fish(
    ca_labels: pd.DataFrame,
    fish_row: int = 0,
    sweep_row: int = 1,
) -> dict[str, str]:
    """Return a mapping from sweep name to fish ID."""
    sweeps = ca_labels.iloc[sweep_row].astype(str).to_numpy()
    fish = ca_labels.iloc[fish_row].astype(str).to_numpy()
    return {sweep: f"fish{fish_id}" for sweep, fish_id in zip(sweeps, fish)}


def apply_event_time_shift(event_frame: int, fish: str, fs: float) -> int:
    """Apply an optional fish-specific event-time correction."""
    shift_s = float(EVENT_TIME_SHIFT_S.get(fish, 0.0))
    return int(round(event_frame + shift_s * fs))


def event_window_bounds(
    event_frame: int,
    n_total_frames: int,
    fs: float,
) -> tuple[int | None, int | None]:
    """Return the start and end frame of the peri-event analysis window."""
    pre_frames = int(round(PRE_S * fs))
    post_frames = int(round(POST_S * fs))

    f0 = event_frame - pre_frames
    f1 = event_frame + post_frames

    if REQUIRE_FULL_EVENT_WINDOW and (f0 < 0 or f1 > n_total_frames):
        return None, None

    f0 = max(f0, 0)
    f1 = min(f1, n_total_frames)
    return int(f0), int(f1)


def valid_neuron_mask(block: np.ndarray) -> np.ndarray:
    """Identify neurons that are finite and non-flat within an event window."""
    block = np.asarray(block, dtype=float)
    return np.all(np.isfinite(block), axis=1) & (np.nanstd(block, axis=1) > 0)


def zscore_rows(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score each valid neuron within one event window.

    Returns
    -------
    z : ndarray, shape (valid_neurons, frames)
    mask : boolean ndarray, shape (original_neurons,)
    """
    block = np.asarray(block, dtype=float)
    mask = valid_neuron_mask(block)
    clean = block[mask]

    if clean.shape[0] == 0:
        return np.empty((0, block.shape[1]), dtype=float), mask

    mean = np.mean(clean, axis=1, keepdims=True)
    sd = np.std(clean, axis=1, keepdims=True)
    z = (clean - mean) / sd
    return z, mask


def threshold_population_traces(
    block: np.ndarray,
    z_threshold: float = Z_THRESHOLD,
) -> dict[str, np.ndarray | int]:
    """
    Compute frame-wise fraction active and raw pair co-activity.

    Parameters
    ----------
    block
        Calcium traces with shape neurons x frames.
    z_threshold
        A neuron is active when its within-window z-score exceeds this value.

    Returns
    -------
    dict with:
        fraction_active : k(t) / N
        coactivity      : k(t)(k(t)-1) / [N(N-1)]
        n_active        : k(t)
        n_neurons       : N after removing invalid/non-flat neurons
    """
    block = np.asarray(block, dtype=float)
    n_frames = block.shape[1] if block.ndim == 2 else 0

    if block.ndim != 2:
        raise ValueError("block must have shape neurons x frames")

    z, _ = zscore_rows(block)
    n_neurons = z.shape[0]

    if n_neurons < MIN_NEURONS or n_frames < MIN_FRAMES:
        nan_trace = np.full(n_frames, np.nan)
        return {
            "fraction_active": nan_trace,
            "coactivity": nan_trace.copy(),
            "n_active": nan_trace.copy(),
            "n_neurons": n_neurons,
        }

    active = z > z_threshold
    k = active.sum(axis=0).astype(float)

    fraction_active = k / n_neurons
    coactivity = (k * (k - 1)) / (n_neurons * (n_neurons - 1))

    return {
        "fraction_active": fraction_active,
        "coactivity": coactivity,
        "n_active": k,
        "n_neurons": n_neurons,
    }


# ============================================================================
# Per-sweep extraction
# ============================================================================
def process_sweep(
    sweep: str,
    fish: str,
    ca_traces: pd.DataFrame,
    motor_events: pd.DataFrame,
    stimulus_onsets: pd.DataFrame,
    fs: float,
) -> list[dict[str, Any]]:
    """Extract event-aligned frame-wise activity measures from one sweep."""
    x = ca_traces.to_numpy(dtype=float).T  # neurons x time

    if x.shape[0] < MIN_NEURONS:
        return []

    n_total_frames = x.shape[1]
    post_frames = int(round(POST_S * fs))
    min_separation = int(round((PRE_S + POST_S) * fs / 2))

    records: list[dict[str, Any]] = []
    window_counter = 0

    def add_event(event_frame: int, condition: str, stimulus: str) -> None:
        nonlocal window_counter

        f0, f1 = event_window_bounds(event_frame, n_total_frames, fs)
        if f0 is None or f1 is None or (f1 - f0) < MIN_FRAMES:
            return

        block = x[:, f0:f1]
        traces = threshold_population_traces(block)
        n_neurons = int(traces["n_neurons"])

        fraction_active = np.asarray(traces["fraction_active"], dtype=float)
        coactivity = np.asarray(traces["coactivity"], dtype=float)
        n_active = np.asarray(traces["n_active"], dtype=float)

        window_id = f"{sweep}_{window_counter:04d}"

        for relative_index in range(f1 - f0):
            frame = f0 + relative_index
            records.append(
                {
                    "fish": fish,
                    "sweep": sweep,
                    "stim": stimulus,
                    "cond": condition,
                    "window_id": window_id,
                    "event_frame": int(event_frame),
                    "event_time_s": float(event_frame / fs),
                    "frame": int(frame),
                    "rel_frame": int(frame - event_frame),
                    "time_s": float(frame / fs),
                    "rel_time_s": float((frame - event_frame) / fs),
                    "n_neurons": n_neurons,
                    "n_active": (
                        float(n_active[relative_index])
                        if np.isfinite(n_active[relative_index])
                        else np.nan
                    ),
                    "fraction_active": (
                        float(fraction_active[relative_index])
                        if np.isfinite(fraction_active[relative_index])
                        else np.nan
                    ),
                    "coactivity": (
                        float(coactivity[relative_index])
                        if np.isfinite(coactivity[relative_index])
                        else np.nan
                    ),
                }
            )

        window_counter += 1

    # Visual stimulus windows
    for stimulus in stimulus_onsets.columns:
        raw_event_frame = int(round(stimulus_onsets[stimulus].item() * fs))
        event_frame = apply_event_time_shift(raw_event_frame, fish, fs)

        if not 0 <= event_frame < n_total_frames:
            continue

        post_f1 = min(event_frame + post_frames, n_total_frames)
        if stimulus in motor_events.columns:
            swam = motor_events[stimulus].iloc[event_frame:post_f1].sum() > 0
        else:
            swam = False

        condition = "visual_swim" if swam else "visual_noswim"
        add_event(event_frame, condition, stimulus)

    # Spontaneous swim windows
    if "spontaneous" in motor_events.columns:
        active_frames = np.flatnonzero(motor_events["spontaneous"].to_numpy() > 0)

        if active_frames.size:
            is_onset = np.concatenate(
                [[True], np.diff(active_frames) > max(min_separation, 1)]
            )
            onset_frames = active_frames[is_onset]

            for raw_event_frame in onset_frames:
                event_frame = apply_event_time_shift(int(raw_event_frame), fish, fs)
                if 0 <= event_frame < n_total_frames:
                    add_event(event_frame, "spont_swim", "spontaneous")

    return records


# ============================================================================
# Plotting
# ============================================================================
def plot_fish_average_timecourse(
    trace_df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    """
    Plot event means within fish, followed by the mean and SEM across fish.

    This avoids treating individual events as independent biological replicates.
    """
    if trace_df.empty or value_column not in trace_df.columns:
        print(f"No data available for {value_column}; figure skipped.")
        return

    event_means = (
        trace_df.groupby(
            ["fish", "cond", "window_id", "rel_time_s"],
            as_index=False,
        )[value_column]
        .mean()
    )

    fish_means = (
        event_means.groupby(["fish", "cond", "rel_time_s"], as_index=False)[
            value_column
        ]
        .mean()
    )

    group_summary = (
        fish_means.groupby(["cond", "rel_time_s"])[value_column]
        .agg(mean="mean", sem="sem", n_fish="count")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for condition in COND_ORDER:
        condition_data = group_summary[group_summary["cond"] == condition].sort_values(
            "rel_time_s"
        )
        if condition_data.empty:
            continue

        t = condition_data["rel_time_s"].to_numpy()
        mean = condition_data["mean"].to_numpy()
        sem = condition_data["sem"].fillna(0).to_numpy()

        ax.plot(
            t,
            mean,
            label=COND_LABELS[condition],
            color=COND_COLORS[condition],
            linewidth=2,
        )
        ax.fill_between(
            t,
            mean - sem,
            mean + sem,
            color=COND_COLORS[condition],
            alpha=0.2,
            linewidth=0,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from event onset (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")


# ============================================================================
# Main
# ============================================================================
def main() -> pd.DataFrame:
    """Run the complete z-score-thresholded population activity analysis."""
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    fs = float(pd.read_csv(Config.ca_sampling_rate_file)["sampling_rate"].item())
    ca_labels = pd.read_csv(Config.ca_labels_file)

    rois_per_sweep = get_rois_per_sweep(ca_labels)
    vr_binaries = load_hdf5_as_dict(Config.vr_all_binaries_file)
    sweep_to_fish = get_sweep_to_fish(ca_labels)

    os.makedirs(OUT_DIR, exist_ok=True)

    all_records: list[dict[str, Any]] = []

    for sweep in rois_per_sweep:
        fish = sweep_to_fish[sweep]
        print(f"Processing {sweep} ({fish})")

        sweep_rois = get_rois_of_one_sweep(sweep, rois_per_sweep)
        sweep_ca_traces = ca_df_f[sweep_rois]
        motor_events = vr_binaries[sweep]

        stimulus_onsets = pd.read_csv(
            f"{Config.BASE_DIR}/stimulus_traces/{sweep}_stimulus_onsets.csv",
            index_col=0,
        )

        all_records.extend(
            process_sweep(
                sweep=sweep,
                fish=fish,
                ca_traces=sweep_ca_traces,
                motor_events=motor_events,
                stimulus_onsets=stimulus_onsets,
                fs=fs,
            )
        )

    trace_df = pd.DataFrame(all_records)

    if EXCLUDE_FISH and not trace_df.empty:
        trace_df = trace_df[~trace_df["fish"].isin(EXCLUDE_FISH)].copy()

    trace_df.to_csv(TRACE_CSV, index=False)
    print(f"Saved {len(trace_df)} frame-level rows: {TRACE_CSV}")

    plot_fish_average_timecourse(
        trace_df=trace_df,
        value_column="fraction_active",
        ylabel="Fraction of active neurons",
        title=f"Z-score-thresholded population recruitment (z > {Z_THRESHOLD:g})",
        save_path=FRACTION_ACTIVE_FIG,
    )

    plot_fish_average_timecourse(
        trace_df=trace_df,
        value_column="coactivity",
        ylabel="Fraction of jointly active neuron pairs",
        title=f"Z-score-thresholded population co-activity (z > {Z_THRESHOLD:g})",
        save_path=COACTIVITY_FIG,
    )

    return trace_df


if __name__ == "__main__":
    main()

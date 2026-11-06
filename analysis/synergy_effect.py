#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Linear Scoring Data for GLMM in R (synergy effects) and compute Selectivity Indices per Neuron
"""
import numpy as np
import pandas as pd
from config import Config


def hierarchical_bootstrap_fast(
    df, metric, n_boot=10000, statfunc=np.mean, null_value=0, random_state=None
):
    """
    Hierarchical bootstrap for ROI metrics nested in animals.
    df must have columns: ["fish", metric]

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ["fish", metric].
    metric : str
        Column to bootstrap.
    n_boot : int
        Number of bootstrap samples.
    statfunc : callable
        Statistic function applied within each animal (default: np.mean).
    null_value : float
        Value to compare against for p-value.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    est : float
        Direct hierarchical point estimate (mean across fish means).
    ci_low, ci_high : float
        Bootstrap confidence interval (2.5%, 97.5%).
    p : float
        One-sided bootstrap p-value vs null_value.
    p_two : float
        Two-sided bootstrap p-value vs null_value.
    boot_stats : np.ndarray
        Bootstrap distribution of estimates.
    """

    rng = np.random.default_rng(random_state)

    # --- Precompute per-animal data ---
    grouped = df.groupby("fish")[metric].apply(lambda x: x.dropna().values)
    animals = grouped.index.to_numpy()
    values_per_animal = grouped.to_dict()

    # --- Direct point estimate ---
    animal_means = [statfunc(vals) for vals in values_per_animal.values() if len(vals) > 0]
    est = np.mean(animal_means)

    # --- Bootstrap distribution ---
    n_animals = len(animals)
    boot_stats = np.empty(n_boot)

    for i in range(n_boot):
        resampled_animals = rng.choice(animals, size=n_animals, replace=True)
        resampled_means = []
        for animal in resampled_animals:
            roi_vals = values_per_animal[animal]
            if len(roi_vals) == 0:
                continue
            roi_resample = rng.choice(roi_vals, size=len(roi_vals), replace=True)
            resampled_means.append(statfunc(roi_resample))
        boot_stats[i] = np.mean(resampled_means) if resampled_means else np.nan

    boot_stats = boot_stats[~np.isnan(boot_stats)]  # drop empty cases (rare)

    # --- CI ---
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])

    # --- p-values ---
    if est > null_value:
        p = np.mean(boot_stats <= null_value)
    else:
        p = np.mean(boot_stats >= null_value)

    p_two = np.mean(np.abs(boot_stats - null_value) >= np.abs(est - null_value))

    return est, ci_low, ci_high, p, p_two, boot_stats


def compute_bootstrap(df_metrics, save_dir):
    # Bootstrap Statistics (Hierarchical Bootstrapping)
    boot_seed = 42
    n_boot = 10000
    boot_summary = list()
    boot_dists = pd.DataFrame()

    metric_name = "vis_spont_index"
    h0 = 0
    est, lo, hi, p, p_two, boot_stats = hierarchical_bootstrap_fast(df_metrics, metric=metric_name, null_value=h0,
                                                                    n_boot=n_boot, random_state=boot_seed)
    boot_summary.append([metric_name, est, lo, hi, p, h0])
    boot_dists[metric_name] = boot_stats

    metric_name = "mixed_spont_index"
    h0 = 0
    est, lo, hi, p, p_two, boot_stats = hierarchical_bootstrap_fast(df_metrics, metric=metric_name, null_value=h0,
                                                                    n_boot=n_boot, random_state=boot_seed)
    boot_summary.append([metric_name, est, lo, hi, p, h0])
    boot_dists[metric_name] = boot_stats

    metric_name = "mixed_vis_index"
    h0 = 0
    est, lo, hi, p, p_two, boot_stats = hierarchical_bootstrap_fast(df_metrics, metric=metric_name, null_value=h0,
                                                                    n_boot=n_boot, random_state=boot_seed)
    boot_summary.append([metric_name, est, lo, hi, p, h0])
    boot_dists[metric_name] = boot_stats

    metric_name = "synergy"
    h0 = 1
    est, lo, hi, p, p_two, boot_stats = hierarchical_bootstrap_fast(df_metrics, metric=metric_name, null_value=h0,
                                                                    n_boot=n_boot, random_state=boot_seed)
    boot_summary.append([metric_name, est, lo, hi, p, h0])
    boot_dists[metric_name] = boot_stats

    metric_name = "synergy_index"
    h0 = 0
    est, lo, hi, p, p_two, boot_stats = hierarchical_bootstrap_fast(df_metrics, metric=metric_name, null_value=h0,
                                                                    n_boot=n_boot, random_state=boot_seed)
    boot_summary.append([metric_name, est, lo, hi, p, h0])
    boot_dists[metric_name] = boot_stats

    boot_summary_df = pd.DataFrame(boot_summary, columns=['metric', 'mean', 'ci_low', 'ci_high', 'p-val', 'h0'])

    boot_summary_df.to_csv(f'{save_dir}/selectivity_indices_bootstrap_statistics.csv')
    boot_dists.to_csv(f'{save_dir}/selectivity_indices_bootstrap_dists.csv')


def prepare_data():
    # ------------------------------------------------------------------------------------------------------------------
    # Load Data
    scores_org = pd.read_csv(f'{Config.BASE_DIR}/data/linear_scoring_results.csv')

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare Data
    # There are mean values for these already present, so we remove them
    idx = scores_org['reg'].isin(
        ['motor_spontaneous', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor',
         'moving_target_02_motor'])
    scores = scores_org[~idx].reset_index(drop=True)

    # Remove _MEAN tag
    scores["reg"] = scores["reg"].str.replace("_MEAN", "", regex=False)

    # Get the unique stimulus names
    stim_names = scores["reg"][~scores["reg"].str.contains("motor")].unique()

    # ------------------------------------------------------------------------------------------------------------------
    # Classify scores as purely visual (stim_only) and motor biased (mixed)

    all_scores = []  # list to hold each stim_score
    # Loop over stimulus names
    for s_name in stim_names:
        # Get all stimulus scores for this stimulus
        stim_score = scores[scores['reg'] == s_name].copy().reset_index(drop=True)

        # Get all motor scores for this stimulus
        motor_score = scores[scores['reg'] == f'{s_name}_motor'].copy().reset_index(drop=True)
        # Find significant motor scores: These are motor biased scores (mixed)
        # idx_motor = motor_score['score'] >= score_th
        idx_motor = motor_score['significant']

        # Give the scores a label (stim_only or stim+motor)
        stim_score['condition'] = 'stim_only'
        stim_score.loc[idx_motor, 'condition'] = 'stim_motor'
        stim_score['stim_name'] = s_name
        all_scores.append(stim_score)

    # Collect stimuli scores
    all_scores_df = pd.concat(all_scores, ignore_index=True)

    # Get scores for spontaneous motor events (motor-only)
    sp_motor_scores = scores[scores['reg'] == 'motor_spontaneous'].copy()
    sp_motor_scores['condition'] = 'motor_only'
    sp_motor_scores['stim_name'] = 'motor_spontaneous'

    # Combine everything in one data frame
    data_set = pd.concat([all_scores_df, sp_motor_scores]).reset_index(drop=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare for Linear Mixed Effects Model
    df = data_set[
        ['score', 'condition', 'stim_name', 'fish', 'roi', 'ca_auc', 'score_norm99', 'significant']].copy()

    # Set all negative scores to 0, since negative scores are "meaningless"
    df.loc[df['score'] < 0, 'score'] = 0

    # Enforce categorical data type
    df['condition'] = pd.Categorical(
        df['condition'],
        categories=['stim_only', 'stim_motor', 'motor_only']
    )

    # Remove all non-significant stimulus scores, we are only interested in real responses (remove non-responders)
    idx_sign = df['significant']
    df = df[idx_sign]

    # Store Data to Disk for R
    df.to_csv(r'D:\WorkingData\PrTecDA_Data\PrDA_somas_Ca_imaging\R_data\stim_motor_data_synergy.csv')
    print('Data store to HDD')


def selectivity_indices():
    # ------------------------------------------------------------------------------------------------------------------
    # Prepare Data
    save_dir = f'{Config.BASE_DIR}/R_data'
    scores_org = pd.read_csv(f'{Config.BASE_DIR}/data/linear_scoring_results.csv')

    # Remove not needed entries from score data set
    idx = scores_org['reg'].isin(['motor_spontaneous_MEAN', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor', 'moving_target_02_motor'])
    df_scores = scores_org[~idx].reset_index(drop=True)

    # Get spontaneous swim events data
    df_sp = df_scores[df_scores['reg'] == 'motor_spontaneous'].copy().reset_index(drop=True)
    df_sp["event_idx"] = df_sp.groupby("roi").cumcount()

    # ------------------------------------------------------------------------------------------------------------------
    # responses to spont. events vs stimulus vs mixed in one roi
    results = []

    # Loop over ROIs (neurons) to get the scores
    for roi in df_scores['roi'].unique():
        roi_data = df_scores[df_scores['roi'] == roi]
        idx = roi_data['reg'] == 'motor_spontaneous'

        # Get significant scores
        sign_sp_idx = roi_data[idx]['significant']

        sign_sp_motor = int(sign_sp_idx.sum())
        if sign_sp_motor >= 3:
            # mean spontaneous motor score
            mean_sp_score = roi_data[idx][sign_sp_idx]['score'].mean()
            mean_sp_auc = roi_data[idx][sign_sp_idx]['ca_auc'].mean()
            mean_sp_norm = roi_data[idx][sign_sp_idx]['score_norm99'].mean()

            # stim data (non-spont)
            idx = roi_data['reg'] != 'motor_spontaneous'
            roi_stim = roi_data[idx][roi_data[idx]['significant']]

            # collapse reg names to either "_motor" or not
            base = roi_stim['reg'].str.replace("_motor$", "", regex=True)
            stim_names = roi_stim['reg'].groupby(base).apply(
                lambda x: x[x.str.endswith("_motor")].iloc[0] if (x.str.endswith("_motor")).any() else x.iloc[0]
            ).values

            has_pure_vis = any(not x.endswith("_motor") for x in stim_names)
            has_motor = any(x.endswith("_motor") for x in stim_names)

            # Take only ROIs that have motor_only, visual_only and mixed scores
            if has_pure_vis and has_motor:
                stim_scores = {stim: roi_stim[roi_stim['reg'] == stim]['score'].item()
                               for stim in stim_names}

                # split motor vs vis
                motor_scores = [v for k, v in stim_scores.items() if k.endswith("_motor")]
                vis_scores = [v for k, v in stim_scores.items() if not k.endswith("_motor")]
                mixed = np.mean(motor_scores) if motor_scores else np.nan
                vis = np.mean(vis_scores) if vis_scores else np.nan
                spont = mean_sp_score
                fish_id = int(roi_data['fish'].unique()[0])

                # also get area under curve of the ca response
                stim_auc = {stim: roi_stim[roi_stim['reg'] == stim]['ca_auc'].item() for stim in stim_names}
                motor_auc = [v for k, v in stim_auc.items() if k.endswith("_motor")]
                vis_auc = [v for k, v in stim_auc.items() if not k.endswith("_motor")]
                mixed_auc_mean = np.mean(motor_auc) if motor_auc else np.nan
                vis_auc_mean = np.mean(vis_auc) if vis_auc else np.nan

                # also get norm. scores
                stim_norm = {stim: roi_stim[roi_stim['reg'] == stim]['score_norm99'].item() for stim in stim_names}
                motor_norm = [v for k, v in stim_norm.items() if k.endswith("_motor")]
                vis_norm = [v for k, v in stim_norm.items() if not k.endswith("_motor")]
                mixed_norm_mean = np.mean(motor_norm) if motor_norm else np.nan
                vis_norm_mean = np.mean(vis_norm) if vis_norm else np.nan

                metrics = {
                    "roi": roi,
                    "fish": fish_id,
                    "vis": vis,  # mean vis score
                    "mixed": mixed,  # mean mixed score
                    "spont": spont,  # mean spont score
                    "delta_vis": vis - spont,
                    "delta_mixed": mixed - spont,
                    "delta_mixed_vis": mixed - vis,
                    "vis/spont": vis / spont if spont else np.nan,
                    "mixed/spont": mixed / spont if spont else np.nan,
                    "mixed/vis": mixed / vis if vis else np.nan,
                    "vis_spont_index": (vis - spont) / (vis + spont),
                    "mixed_spont_index": (mixed - spont) / (mixed + spont),
                    "mixed_vis_index": (mixed - vis) / (mixed + vis),
                    "synergy_index": (mixed - (vis + spont)) / (mixed + vis + spont),
                    "synergy": mixed / (vis + spont) if (vis + spont) else np.nan,
                    "spont_auc": mean_sp_auc,
                    "vis_auc": vis_auc_mean,
                    "mixed_auc": mixed_auc_mean,
                    "spont_norm": mean_sp_norm,
                    "vis_norm": vis_norm_mean,
                    "mixed_norm": mixed_norm_mean
                }

                results.append(metrics)

    # Convert to DataFrame
    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(f'{save_dir}/score_metrics_per_neuron.csv')
    print('Selectivity Indices stored to HDD')


def main():
    # Prepare Linear Scoring Data for Generalized Linear Mixed Model in R (synergy effect)
    prepare_data()

    # Compute Selectivity Indices based on scores for visual, motor and both (per Neuron)
    # then store to HDD
    selectivity_indices()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comment
"""
import numpy as np
import pandas as pd
from utils import load_hdf5_as_dict, calcium_impulse_response, get_rois_per_sweep, create_regressors_from_binary, \
    get_rois_of_one_sweep, save_dat_frame_as_hdf5
from config import Config
import statsmodels.api as sm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.integrate import simpson
from scipy.signal import fftconvolve

"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'linear_scoring_analysis.py'

Score the response of ROIs to different stimuli and motor events using a linear regression model.
Score = R² * slope


Result: 
    - /data/linear_scoring_results.csv'
...

Requirements:


Nils Brehm  -  2025
"""


def get_spontaneous_events(vr_events, rois_per_sweep):
    sp_events = dict()
    number_sp_events = pd.DataFrame()
    rois_with_sp = 0
    for sw in vr_events:
        idx = vr_events[sw]['stimulus'] == 0
        sp_events[sw] = vr_events[sw][idx]
        event_count = sp_events[sw].shape[0]
        if event_count > 0:
            rois_with_sp = rois_with_sp + rois_per_sweep[sw].shape[0]
        number_sp_events[sw] = [event_count]

        # print(f'{sw}: {sp_events[sw].shape[0]} spont. swims')
    return sp_events, number_sp_events, rois_with_sp


def find_best_lag_cross_correlation(data_trace, design_matrix, lag_start, lag_end, plot=False):
    # lag_start: positive number (allowed shifts to the left)
    # lag_end: positive number (allowed shifts to the right)

    corr = np.correlate(data_trace, design_matrix.iloc[:, 1], mode='full')
    # Number of samples in the original signals
    n = len(data_trace)  # Assumes both signals are the same length

    # Create an array of lag values. Lags range from -(n-1) to (n-1)
    corr_lags = np.arange(-n + 1, n)

    # Find the index of the maximum correlation value
    max_index = np.argmax(corr)

    corr_best_lag = corr_lags[max_index]

    # If the best lag is in the defined window, take it
    if -lag_start < corr_best_lag <= lag_end:
        result = corr_best_lag
    elif corr_best_lag < -lag_start:
        result = -lag_start
    elif corr_best_lag > lag_end:
        result = lag_end
    else:
        result = 0

    if plot:
        fig, axs = plt.subplots()
        axs.plot(corr_lags, corr, marker='o')
        axs.set_xlabel('Lag')
        axs.set_ylabel('Correlation')
        axs.axvline(x=corr_best_lag, color='r', linestyle='--', label=f'Best lag = {corr_best_lag}')
        axs.legend()
        plt.show()

    return result, corr_best_lag


def merge_events_closing(trace, max_gap):
    from scipy.ndimage import binary_closing

    trace = np.array(trace)
    # Create a structuring element that covers gaps of size max_gap
    structure = np.ones(max_gap * 2 + 1)

    # binary_closing will fill gaps that are smaller than the structuring element
    closed = binary_closing(trace, structure=structure)
    return closed.astype(int)


def get_event_onset_offset(data):
    """
    Detects the onset and offset indices of events in a merged binary trace.

    Parameters:
        data (array-like): Binary trace (numpy array or list of 0's and 1's).

    Returns:
        onsets (numpy.ndarray): Indices where events begin.
        offsets (numpy.ndarray): Indices where events end.
    """
    data = np.asarray(data)
    # Compute the difference between consecutive elements
    d = np.diff(data.astype(int))

    # Onset: Transition from 0 to 1 (difference equals +1)
    onsets = np.where(d == 1)[0] + 1  # +1 to account for the shift due to diff

    # Offset: Transition from 1 to 0 (difference equals -1)
    offsets = np.where(d == -1)[0]

    # If the signal starts with a 1, then index 0 is an onset.
    if data[0] == 1:
        onsets = np.insert(onsets, 0, 0)

    # If the signal ends with a 1, then the last index is an offset.
    if data[-1] == 1:
        offsets = np.append(offsets, len(data) - 1)

    return onsets, offsets


def linear_regression_model(data_trace, design_matrix, lag, debugging=False):
    # Shift the regressors by the current lag.
    # Note: The shift() function shifts the index by the given number of periods.
    x_shifted = design_matrix.shift(lag)

    # Combine y and shifted X into a single DataFrame and drop rows with NaNs.
    df = pd.concat([data_trace, x_shifted], axis=1).dropna()
    # df = pd.concat([data_trace, x_shifted], axis=1).fillna(0)

    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1:]

    # Compute LM-OLS Model
    model = sm.OLS(y_clean, x_clean).fit()
    r2 = model.rsquared
    cf = model.params.iloc[1]

    score = cf * r2
    # if score < 0:
    #     score = 0
    # if r2 <= 0:
    #     score = 0
    # if cf < 0:
    #     score = 0

    scoring_results = {
        'r2': r2,
        'cf': float(cf),
        'score': float(score)
    }
    if debugging:
        return scoring_results, x_clean, y_clean
    else:
        return scoring_results


def plot_motor_regs(bouts, events, ca_trace, motor_spontaneous, motor_spontaneous_events):
    # LM
    design_matrix = sm.add_constant(bouts)
    # Compute LM-OLS Model
    model = sm.OLS(ca_trace, design_matrix).fit()
    r2_b = model.rsquared
    int_b = model.params[0]
    cf_b = model.params[1]
    score_bouts = cf_b * r2_b
    print(f'Bouts: {score_bouts} (r2={r2_b:.3f}, slope:{cf_b:.3f})')

    design_matrix = sm.add_constant(events)
    # Compute LM-OLS Model
    model = sm.OLS(ca_trace, design_matrix).fit()
    r2_e = model.rsquared
    int_e = model.params[0]
    cf_e = model.params[1]
    score_events = cf_e * r2_e
    print(f'Events: {score_events} (r2={r2_e:.3f}, slope:{cf_e:.3f})')

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(bouts)
    axs[0].plot(ca_trace, 'k', alpha=0.3)
    axs[0].plot(motor_spontaneous, 'g', alpha=0.3)
    axs[0].set_title('Bouts')
    axs[1].plot(events)
    axs[1].plot(ca_trace, 'k', alpha=0.3)
    axs[1].plot(motor_spontaneous_events, 'g', alpha=0.3)
    axs[1].set_title('Events')
    plt.tight_layout()
    plt.show()


def linear_scoring(ca_data, motor_events, visual_events, ca_sampling_rate, stimulus_onsets, stimulus_offsets, settings):
    # Generate Calcium Impulse Response Function
    _, cif = calcium_impulse_response(tau_rise=settings['tau_rise'], tau_decay=settings['tau_decay'], amplitude=1.0,
                                      sampling_rate=ca_sampling_rate, threshold=1e-5, norm=True)

    # Dummy code for strange cases
    dummy_code = -1
    max_gap_secs = settings['max_gap_secs']

    # =======================================================================
    # Spontaneous Motor Events
    # Collect Motor Events
    motor_spontaneous = motor_events['spontaneous']

    # Spontaneous Motor Events: Merge Motor Events that are too close (for later cutting out traces)
    motor_spontaneous_events = merge_events_closing(motor_spontaneous, max_gap=int(max_gap_secs * ca_sampling_rate))

    # Find onset and offsets of  spontaneous motor events
    motor_spontaneous_idx = get_event_onset_offset(motor_spontaneous_events)

    # Generate calcium regressors (visual, motor and spontaneous motor)
    # Compute the Regressor for all events, not the merged ones:
    # motor_spontaneous_regressor = create_regressors_from_binary(motor_spontaneous.to_numpy(), cif, delta=True, norm=False)
    # Compute the Regressor for merged events:
    motor_spontaneous_regressor = create_regressors_from_binary(motor_spontaneous_events, cif, delta=True, norm=False)

    # # Plot Bouts vs Merged Events Regs
    # bouts = create_regressors_from_binary(motor_spontaneous.to_numpy(), cif, delta=True, norm=False)
    # events = create_regressors_from_binary(motor_spontaneous_events, cif, delta=True, norm=False)
    # plot_motor_regs(bouts, events, ca_data.iloc[:, 10].to_numpy(), motor_spontaneous, motor_spontaneous_events)

    # =======================================================================
    # Visual + Motor Regs
    visual_regressors = pd.DataFrame()
    motor_stimulus_regressors = pd.DataFrame()
    for k in visual_events:
        # Visual Regressor
        visual_regressors[k] = create_regressors_from_binary(visual_events[k], cif, delta=True, norm=True)

        # Motor Regressor for each stimulus: Merge Motor Events that are too close
        motor_merged = merge_events_closing(motor_events[k], max_gap=int(max_gap_secs * ca_sampling_rate))
        motor_stimulus_regressors[k] = create_regressors_from_binary(motor_merged, cif, delta=True, norm=True)

    # =======================================================================
    sweep_scoring_result = list()

    # Loop over each ROI (column in df)
    for roi in ca_data.columns:
        # roi = '20'
        scores_list = list()
        r2_list = list()
        cf_list = list()
        reg_list = list()
        lags_list = list()
        reg_area_list = list()
        ca_area_list = list()

        y_trace = ca_data[roi]  # Get calcium trace
        # plt.plot(motor_spontaneous_regressor)
        # plt.plot(y_trace)
        # plt.show()

        # Linear Scoring for Visual Stimulus regressors
        for s_k, e_k in zip(stimulus_onsets, stimulus_offsets):
            # s_k and e_k are both the stimulus name (key)
            # s_k = 'dark_loom'
            # e_k = 'dark_loom'
            # ==========================================================================================================
            # VISUAL REGRESSORS
            time_before = settings['time_window'][s_k][0]
            time_after = settings['time_window'][s_k][1]
            lag_start_sec = settings['lag_window'][s_k][0]
            lag_end_sec = settings['lag_window'][s_k][1]

            # Create window for cutting out
            s_idx = int((stimulus_onsets[s_k].item() - time_before) * ca_sampling_rate)
            e_idx = int((stimulus_offsets[e_k].item() + time_after) * ca_sampling_rate)

            # Cut out stimulus window
            y_segment = y_trace[s_idx:e_idx].reset_index(drop=True)
            reg = visual_regressors[s_k][s_idx:e_idx].reset_index(drop=True)

            # Create design matrix and add Intercept
            design_matrix = sm.add_constant(reg)

            # Linear Regression with Time Alignment (Cross Correlation)
            best_lag, cbl = find_best_lag_cross_correlation(
                y_segment, design_matrix, int(lag_start_sec*ca_sampling_rate), int(lag_end_sec*ca_sampling_rate)
            )
            lm_result = linear_regression_model(y_segment, design_matrix, best_lag)

            best_lag_secs = best_lag / ca_sampling_rate
            scores_list.append(lm_result['score'])
            r2_list.append(lm_result['r2'])
            cf_list.append(lm_result['cf'])
            reg_list.append(s_k)
            lags_list.append(best_lag_secs)

            # Get area under the regressor as a metrics for motor strength
            t_segment = np.arange(len(y_segment)) / ca_sampling_rate
            reg_area = simpson(reg.to_numpy(), t_segment)  # Simpson's rule
            y_area = simpson(y_segment, t_segment)  # Ca response area
            reg_area_list.append(reg_area)
            ca_area_list.append(y_area)

            # ==========================================================================================================
            # MOTOR EVENTS
            # Look for Motor Events during Stimulus Window
            motor_reg = motor_stimulus_regressors[s_k][s_idx:e_idx].reset_index(drop=True)

            # Create design matrix and add Intercept
            motor_design_matrix = sm.add_constant(motor_reg)

            # Linear Regression with Time Alignment
            motor_best_lag, cbl = find_best_lag_cross_correlation(
                y_segment, motor_design_matrix,
                int(lag_start_sec*ca_sampling_rate), int(lag_end_sec*ca_sampling_rate),
                plot=False
            )
            motor_lm_result = linear_regression_model(y_segment, motor_design_matrix, motor_best_lag)

            motor_best_lag_secs = motor_best_lag / ca_sampling_rate

            scores_list.append(motor_lm_result['score'])
            r2_list.append(motor_lm_result['r2'])
            cf_list.append(motor_lm_result['cf'])
            reg_list.append(f'{s_k}_motor')
            lags_list.append(motor_best_lag_secs)

            # Get area under the motor regressor as a metrics for motor strength
            t_segment = np.arange(len(y_segment)) / ca_sampling_rate
            reg_area = simpson(reg.to_numpy(), t_segment)  # Simpson's rule
            y_area = simpson(y_segment, t_segment)  # Ca response area
            reg_area_list.append(reg_area)
            ca_area_list.append(y_area)

        # ==============================================================================================================
        # Motor Spontaneous Events
        if motor_spontaneous_idx[0].shape[0] > 0:
            for s_idx, e_idx in zip(motor_spontaneous_idx[0], motor_spontaneous_idx[1]):
                # Cut out stimulus window
                time_before = settings['time_window']['motor_spontaneous'][0]
                time_after = settings['time_window']['motor_spontaneous'][1]
                lag_start_sec = settings['lag_window']['motor_spontaneous'][0]
                lag_end_sec = settings['lag_window']['motor_spontaneous'][1]

                start_idx = s_idx - int(time_before * ca_sampling_rate)
                end_idx = e_idx + int(time_after * ca_sampling_rate)

                y_segment = y_trace[start_idx:end_idx].reset_index(drop=True)
                reg_0 = motor_spontaneous_regressor[start_idx:end_idx]
                reg = pd.DataFrame(reg_0)

                # Create design matrix and add Intercept
                design_matrix = sm.add_constant(reg)

                # Linear Regression with Time Alignment
                best_lag, cbl = find_best_lag_cross_correlation(
                    y_segment, design_matrix,
                    int(lag_start_sec * ca_sampling_rate), int(lag_end_sec * ca_sampling_rate), plot=False
                )
                lm_result = linear_regression_model(y_segment, design_matrix, best_lag)

                best_lag_secs = best_lag / ca_sampling_rate
                scores_list.append(lm_result['score'])
                r2_list.append(lm_result['r2'])
                cf_list.append(lm_result['cf'])
                reg_list.append('motor_spontaneous')
                lags_list.append(best_lag_secs)

                # Get area under the spont. motor regressor as a metrics for motor strength
                t_segment = np.arange(len(y_segment)) / ca_sampling_rate
                reg_area = simpson(reg_0, t_segment)  # Simpson's rule
                y_area = simpson(y_segment, t_segment)  # Ca response area
                reg_area_list.append(reg_area)
                ca_area_list.append(y_area)

        # Collect results
        metrics = ['score', 'r2', 'coeff', 'lag_sec', 'ca_auc', 'reg_auc']
        scoring = pd.DataFrame([scores_list, r2_list, cf_list, lags_list, ca_area_list, reg_area_list]).transpose()
        scoring.columns = metrics
        scoring['reg'] = reg_list

        # Moving Targets are presented twice. So take the mean
        idx_1 = scoring['reg'] == 'moving_target_01'
        idx_2 = scoring['reg'] == 'moving_target_02'
        mt_means = pd.DataFrame([scoring[idx_1+idx_2][metrics].mean().to_numpy()], columns=metrics)
        mt_means['reg'] = 'moving_target_MEAN'
        scoring = pd.concat([scoring, mt_means])

        # Same for the Motor Regressor
        idx_1 = scoring['reg'] == 'moving_target_01_motor'
        idx_2 = scoring['reg'] == 'moving_target_02_motor'
        motor_mt_means = pd.DataFrame([scoring[idx_1+idx_2][metrics].mean().to_numpy()], columns=metrics)
        motor_mt_means['reg'] = 'moving_target_motor_MEAN'
        scoring = pd.concat([scoring, motor_mt_means])

        # Motor Spontaneous Mean
        idx_sp = scoring['reg'] == 'motor_spontaneous'
        if motor_spontaneous_idx[0].shape[0] > 0:
            motor_sp_means = pd.DataFrame([scoring[idx_sp][metrics].mean().to_numpy()], columns=metrics)
        else:
            motor_sp_means = pd.DataFrame([np.zeros(len(metrics)) + dummy_code], columns=metrics)

        motor_sp_means['reg'] = 'motor_spontaneous_MEAN'
        scoring = pd.concat([scoring, motor_sp_means])

        # Add ROI name
        scoring['roi'] = roi
        scoring = scoring.reset_index(drop=True)
        sweep_scoring_result.append(scoring)

    results_df = pd.concat(sweep_scoring_result)
    return results_df


def compute_linear_scoring(settings):
    from time import perf_counter
    t0 = perf_counter()
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
    ca_labels = pd.read_csv(Config.ca_labels_file)
    rois_per_sweep = get_rois_per_sweep(ca_labels)
    vr_binaries = load_hdf5_as_dict(Config.vr_all_binaries_file)

    for k in rois_per_sweep:
        print(k)

    all_results = list()
    for sw in rois_per_sweep:
        # sw = 'sw_31'
        # print('FORCED SW!!!')

        # Get all the ROIs of this sweep
        sw_rois = get_rois_of_one_sweep(sw, rois_per_sweep)

        # Get Binaries for all Stimuli
        sw_stimulus_binaries = pd.read_csv(f'{Config.BASE_DIR}/stimulus_traces/{sw}_stimulus_binaries.csv', index_col=0)
        sw_stimulus_onsets = pd.read_csv(f'{Config.BASE_DIR}/stimulus_traces/{sw}_stimulus_onsets.csv', index_col=0)
        sw_stimulus_offsets = pd.read_csv(f'{Config.BASE_DIR}/stimulus_traces/{sw}_stimulus_offsets.csv', index_col=0)

        # Get Spontaneous and Stimulus Motor Events for this sweep
        motor_events = vr_binaries[sw]

        # Get all ca data traces from all ROIs
        sw_ca_traces = ca_df_f[sw_rois]

        print(f'Sweep: {sw}')
        scores = linear_scoring(
            sw_ca_traces, motor_events, sw_stimulus_binaries, ca_sampling_rate,
            sw_stimulus_onsets, sw_stimulus_offsets, settings
        )
        # Add sweep name
        scores['sw'] = sw

        # Add anatomical coordinates
        anatomical_positions_x = list()
        anatomical_positions_y = list()
        z_planes = list()
        for roi in scores['roi'].unique():
            n = scores[scores['roi'] == roi].shape[0]
            pos = ca_labels[roi].iloc[3]
            x_coordinate, y_coordinate = map(int, pos.split('-'))

            anatomical_positions_x.extend([x_coordinate] * n)
            anatomical_positions_y.extend([y_coordinate] * n)

            z_val = int(ca_labels[roi].iloc[4])
            z_planes.extend([z_val] * n)

        scores['x_position'] = anatomical_positions_x
        scores['y_position'] = anatomical_positions_y
        scores['z_position'] = z_planes
        scores = scores.reset_index(drop=True)
        # print(f'This took: {(perf_counter() - t1) / 60:.3f} minutes.')

        # Put everything together
        all_results.append(scores)

    results_df = pd.concat(all_results).reset_index(drop=True)

    # Add Fish IDs to the data
    sw_fish_ids = ca_labels.iloc[[0, 1], :].T.drop_duplicates().reset_index(drop=True)
    sw_fish_ids.columns = ['fish', 'sw']
    results_df = results_df.merge(sw_fish_ids, on="sw", how="left")

    # Normalize scores by the 99th percentile per Sweep (recording)
    results_df["score_norm99"] = results_df.groupby("sw")["score"].transform(
        lambda x: x / x.quantile(0.99)
    )

    # Significance Testing
    try:
        score_th = pd.read_csv(f'{Config.BASE_DIR}/data/scoring_thresholds.csv', index_col=0, names=["roi", "roi_value"], header=0)
        r2_th = pd.read_csv(f'{Config.BASE_DIR}/data/r2_thresholds.csv', index_col=0, names=["roi", "roi_value"], header=0)
        slope_th = pd.read_csv(f'{Config.BASE_DIR}/data/slope_thresholds.csv', index_col=0, names=["roi", "roi_value"], header=0)
        score_th.index = score_th.index.astype(str)
        r2_th.index = r2_th.index.astype(str)
        slope_th.index = slope_th.index.astype(str)

        # map each ROI to its threshold
        results_df["threshold"] = results_df["roi"].map(score_th["roi_value"])
        results_df["threshold_r2"] = results_df["roi"].map(r2_th["roi_value"])
        results_df["threshold_slope"] = results_df["roi"].map(slope_th["roi_value"])

        # results_df["significant"] = results_df["score"] > results_df["threshold"]
        # check if score > threshold or r2 > threshold_r2
        idx_sign = (results_df["score"] >= results_df["threshold"]) | (results_df["r2"] >= results_df["threshold_r2"])
        results_df["significant"] = idx_sign

    except FileNotFoundError:
        print('Could not find null distribution, skip significance test')

    results_df.to_csv(f'{Config.BASE_DIR}/data/linear_scoring_results.csv', index=False)

    print('==== FINISHED LINEAR SCORING ====')
    print(f'This took: {(perf_counter() - t0) / 60:.3f} minutes.')


def single_permutation_vectorized(ca_df_f, cif, ca_sampling_rate, settings, data_size):
    # Random time point
    time_point = np.random.randint(20, data_size - 20)
    time_before_samples = int(settings['time_before'] * ca_sampling_rate)
    time_after_samples = int(settings['time_after'] * ca_sampling_rate)
    start_idx = time_point - time_before_samples
    end_idx = time_point + time_after_samples

    # Build regressor trace (same for all ROIs)
    random_binary = np.zeros(data_size)
    random_binary[time_point] = 1
    reg_trace = create_regressors_from_binary(random_binary, cif, delta=True, norm=True)
    reg = reg_trace[start_idx:end_idx]

    # Cut out Ca segments (all neurons at once)
    Y = ca_df_f[start_idx:end_idx, :]   # shape (time_window, n_rois)

    # Design matrix: constant + regressor
    X = np.column_stack([np.ones_like(reg), reg])  # shape (time_window, 2)

    # Solve OLS via normal equations: betas = (XᵀX)^(-1) XᵀY
    XtX_inv = np.linalg.inv(X.T @ X)
    betas = XtX_inv @ X.T @ Y     # shape (2, n_rois)

    # Predicted values
    Y_hat = X @ betas             # shape (time_window, n_rois)

    # Compute R² per ROI
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot

    # Extract coefficients for regressor (2nd row of betas)
    cf = betas[1, :]

    # Final score = r² * coefficient
    scores = r2 * cf

    return scores, r2, cf


def single_permutation(ca_df_f, cif, ca_sampling_rate, settings, data_size):
    # permutation_round = []
    # permutation_round = np.zeros(ca_df_f.shape[1])  # one per ca_trace
    perm_scores = np.zeros(ca_df_f.shape[1])
    perm_r2 = np.zeros(ca_df_f.shape[1])
    perm_slope = np.zeros(ca_df_f.shape[1])

    # Compute score for one random event for each ROI (neuron) Ca trace
    # for ca_trace in ca_df_f.T:
    for i, ca_trace in enumerate(ca_df_f.T):
        # Get one random time point (as sample idx)
        time_point = np.random.randint(20, data_size - 20)

        # Set cutout window
        time_before_samples = int(settings['time_before'] * ca_sampling_rate)
        time_after_samples = int(settings['time_after'] * ca_sampling_rate)

        # Create binary trace and then regressor (one event)
        random_binary = np.zeros(data_size)
        random_binary[time_point] = 1
        reg_trace = create_regressors_from_binary(random_binary, cif, delta=True, norm=True)

        # Cut out Ca trace and reg
        start_idx = time_point - time_before_samples
        end_idx = time_point + time_after_samples
        y_segment = ca_trace[start_idx:end_idx]
        reg = reg_trace[start_idx:end_idx]

        # Linear Model for Scoring
        design_matrix = sm.add_constant(reg)
        model = sm.OLS(y_segment, design_matrix).fit()
        r2 = model.rsquared
        cf = model.params[1]
        score = r2 * cf
        # permutation_round[i] = float(score)
        perm_scores[i] = score
        perm_r2[i] = r2
        perm_slope[i] = cf

    return perm_scores, perm_r2, perm_slope


def permutation_null_distribution(settings):
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    ca_df_f = ca_df_f.to_numpy()
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
    data_size = ca_df_f.shape[0]

    _, cif = calcium_impulse_response(tau_rise=settings['tau_rise'], tau_decay=settings['tau_decay'],
                                      amplitude=1.0, sampling_rate=ca_sampling_rate, threshold=1e-5, norm=True)

    # ------------------------------------------------------------------------------------------------------------------
    # DEBUGGING
    # r = single_permutation(ca_df_f, cif, ca_sampling_rate, settings, data_size)
    # rr = single_permutation_vectorized(ca_df_f, cif, ca_sampling_rate, settings, data_size)
    # ------------------------------------------------------------------------------------------------------------------

    # # Parallel execution over permutations using (slow) python loop version
    # results = Parallel(n_jobs=-1)(  # Use all cores
    #     delayed(single_permutation)(ca_df_f, cif, ca_sampling_rate, settings, data_size)
    #     for _ in tqdm(settings['n_permutations'])
    # )
    # # result: (n_permutations, metrics, neurons)
    # null_dist = np.array(results)

    # Parallel execution over permutations using (fast) vectorized version
    results = Parallel(n_jobs=-1)(  # Use all cores
        delayed(single_permutation_vectorized)(ca_df_f, cif, ca_sampling_rate, settings, data_size)
        for _ in tqdm(range(settings['n_permutations']))
    )
    # result: (n_permutations, metrics, neurons)
    null_dist = np.array(results)

    dfs = [
        pd.DataFrame(null_dist[:, i, :], columns=[f"{j}" for j in range(null_dist.shape[2])])
        for i in range(null_dist.shape[1])
    ]
    null_score_dist = dfs[0]
    null_r2_dist = dfs[1]
    null_slope_dist = dfs[2]
    return null_score_dist, null_r2_dist, null_slope_dist


def create_regressors_from_binary_batch(binary_events, cif, norm=True):
    n_perms, data_size = binary_events.shape
    kernel = cif.copy()

    if norm:
        kernel = kernel / np.linalg.norm(kernel)

    reg_traces = np.array([
        fftconvolve(binary_events[i], kernel, mode="same")
        for i in range(n_perms)
    ])
    return reg_traces


def batch_permutations_fully_vectorized(ca_df_f, cif, ca_sampling_rate, settings, data_size, n_perms=1000):
    # Random events
    time_points = np.random.randint(20, data_size - 20, size=n_perms)
    time_before = int(settings['time_before'] * ca_sampling_rate)
    time_after = int(settings['time_after'] * ca_sampling_rate)
    win_size = time_before + time_after

    # --- 1. Build binary matrix ---
    binary_events = np.zeros((n_perms, data_size))
    binary_events[np.arange(n_perms), time_points] = 1

    # --- 2. Batch regressors ---
    reg_traces = create_regressors_from_binary_batch(binary_events, cif, norm=True)

    # --- 3. Collect windows (vectorized indexing) ---
    idx = np.arange(win_size)[None, :]  # shape (1, win_size)
    start_idx = (time_points - time_before)[:, None]  # shape (n_perms, 1)
    window_idx = start_idx + idx  # shape (n_perms, win_size)

    # regressor windows
    reg_windows = reg_traces[np.arange(n_perms)[:, None], window_idx]  # (n_perms, win_size)

    # Ca windows
    Y_windows = ca_df_f[window_idx, :]  # (n_perms, win_size, n_rois)

    # --- 4. Build design matrices ---
    ones = np.ones((n_perms, win_size, 1))
    regs = reg_windows[:, :, None]
    X = np.concatenate([ones, regs], axis=2)  # (n_perms, win_size, 2)

    # --- 5. Batched regression ---
    Xt = np.transpose(X, (0, 2, 1))                  # (n_perms, 2, win_size)
    XtX = Xt @ X                                     # (n_perms, 2, 2)
    XtX_inv = np.linalg.inv(XtX)                     # (n_perms, 2, 2)
    XtY = Xt @ Y_windows                             # (n_perms, 2, n_rois)
    betas = XtX_inv @ XtY                            # (n_perms, 2, n_rois)

    # --- 6. Predictions + R² ---
    Y_hat = np.einsum("pwt,ptr->pwr", X, betas)      # (n_perms, win_size, n_rois)
    ss_res = np.sum((Y_windows - Y_hat) ** 2, axis=1)
    ss_tot = np.sum((Y_windows - Y_windows.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1 - ss_res / ss_tot                         # (n_perms, n_rois)

    cf = betas[:, 1, :]                              # regressor coefficient
    scores = r2 * cf                                 # (n_perms, n_rois)

    return scores


def compute_thresholds(null_dist, ca_labels):
    # Get
    thresholds_df = pd.DataFrame([np.percentile(null_dist, 95, axis=0)], columns=ca_labels.columns)
    # extract thresholds row
    thresholds = thresholds_df.iloc[0]
    return thresholds


def compute_null_distributions():
    ca_labels = pd.read_csv(Config.ca_labels_file)

    # Null Distribution (Permutation/Bootstrapping Test)
    permutation_settings = Config.permutation_settings
    null_score_dist, null_r2_dist, null_slope_dist = permutation_null_distribution(permutation_settings)

    # Compute thresholds per neuron
    score_ths = compute_thresholds(null_score_dist, ca_labels)
    r2_ths = compute_thresholds(null_r2_dist, ca_labels)
    slope_ths = compute_thresholds(null_slope_dist, ca_labels)

    # Store to HDD
    save_dat_frame_as_hdf5(f'{Config.BASE_DIR}/data/scores_null_distribution.h5', pd.DataFrame(null_score_dist))
    save_dat_frame_as_hdf5(f'{Config.BASE_DIR}/data/r2_null_distribution.h5', pd.DataFrame(null_r2_dist))
    save_dat_frame_as_hdf5(f'{Config.BASE_DIR}/data/slopes_null_distribution.h5', pd.DataFrame(null_slope_dist))
    score_ths.to_csv(f'{Config.BASE_DIR}/data/scoring_thresholds.csv')
    r2_ths.to_csv(f'{Config.BASE_DIR}/data/r2_thresholds.csv')
    slope_ths.to_csv(f'{Config.BASE_DIR}/data/slope_thresholds.csv')
    print('==== FINISHED PERMUTATION NULL DISTRIBUTION ====')


def main():
    # Compute Statistical Null Distribution
    # compute_null_distributions()

    # Compute Linear Scoring
    settings = Config.linear_scoring_settings
    compute_linear_scoring(settings)


if __name__ == "__main__":
    main()

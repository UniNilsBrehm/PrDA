import numpy as np
import pandas as pd
from utils import get_rois_per_sweep, broaden_binary_trace, binary_to_gaussian, create_regressors_from_binary, \
    calcium_impulse_response, filter_low_pass
from config import Config
from IPython import embed


"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'remove_motor_events_from_ca_responses.py'

Remove Responses to Motor Events from Ca Imaging Traces.


Result: 
    - /data/df_f_data_no_motor.csv'
...

Requirements:


Nils Brehm  -  2025
"""


def mask_ca_trace_by_motor_events(ca_trace, binary, ca_sampling_rate, window):
    """
    Remove Responses to Motor Events from Ca Imaging Traces.


    Parameters:
        ca_trace (data_frame): Ca Imaging Traces. Each Column is a ROI.
        binary (np.ndarray): Binary trace (1D array containing 0s and 1s).
        ca_sampling_rate (float): Sampling Rate in Hz.
        window (float): Expansion Time in seconds.

    Returns:
        pd.DataFrame: Motor corrected Ca Imaging Data.
    """

    # Broaden the binary to make it less sharp in time
    binary_broad = broaden_binary_trace(binary.copy(), width=int(window*ca_sampling_rate))

    # Turn binary rectangles into smooth gaussian shapes
    gaussian_trace = binary_to_gaussian(binary_broad, sampling_rate=ca_sampling_rate, sigma=2)

    # Inverse the smooth binary trace and multiply it by the Ca Imaging Trace to remove responses to Motor Events.
    masked_trace = ca_trace * (1-gaussian_trace)
    return masked_trace


def subtract_motor_regressors(ca_trace, binary, ca_sampling_rate):
    tau_rise = Config.linear_scoring_settings['tau_rise']
    tau_decay = Config.linear_scoring_settings['tau_decay']
    _, cirf = calcium_impulse_response(tau_rise=tau_rise, tau_decay=tau_decay, norm=True, sampling_rate=ca_sampling_rate)

    reg = create_regressors_from_binary(binary, cirf, norm=True)

    # from scipy.ndimage import gaussian_filter1d
    # sigma_sec = 6
    # sigma_samples = int(sigma_sec * ca_sampling_rate)
    # reg = gaussian_filter1d(binary.astype(float), sigma=sigma_samples)

    # Move the reg slightly into the future
    # Shift left by x samples (i.e., events start earlier)
    shift_amount = 4
    reg_shifted = np.roll(reg, -shift_amount)

    # Zero out the last `shift_amount` elements (wrapped-around data)
    reg_shifted[-shift_amount:] = 0

    # Fit regression weight (alpha)
    alpha = np.dot(ca_trace, reg_shifted) / np.dot(reg_shifted, reg_shifted + 1e-8)  # neuron-specific weight

    suppression_factor = 100
    # Add 1 to the reg to avoid division by zero
    # multiply reg with a suppression_factor, this will increase the attenuating effect
    reg_sup = 1 + reg_shifted * suppression_factor * alpha

    # Avoid dividing by zero or negatives
    reg_sup = np.maximum(reg_sup, 1e-3)

    # 4. Divide Ca trace by regressor
    de_motored = ca_trace / reg_sup
    # de_motored = ca_trace - alpha * reg_shifted

    # Lowpass filter the resulting trace to remove sharp edges a bit
    de_motored_fil = filter_low_pass(de_motored, cutoff=0.5, fs=ca_sampling_rate)

    import matplotlib.pyplot as plt
    plt.plot(ca_trace, 'k', label='ca trace')
    plt.plot(reg_shifted, 'r', label='motor reg')
    plt.plot(de_motored_fil, 'tab:orange', label='unbiased smoothed')
    plt.legend()
    plt.show()


    embed()
    exit()

    return de_motored_fil


def main():
    # Directories
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    vr_binaries_dir = f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries.csv'
    ca_labels_file = f'{base_dir}/data/df_f_data_labels.csv'

    ca_sampling_rate_file = f'{base_dir}/meta_data/sampling_rate.csv'
    ca_df_f_file = f'{base_dir}/data/df_f_data.csv'
    ca_labels = pd.read_csv(ca_labels_file)

    window_sec = 10  # time to expand (broaden) the binary in both directions in seconds

    # Load Data
    ca_df_f = pd.read_csv(ca_df_f_file)
    ca_sampling_rate = pd.read_csv(ca_sampling_rate_file)['sampling_rate'].item()

    # Detected VR Events for selected data set as a binary trace
    vr_binaries = pd.read_csv(vr_binaries_dir)

    rois_per_sweep = get_rois_per_sweep(ca_labels)
    results = []
    for sw in rois_per_sweep:
        binary = vr_binaries[sw]
        for roi in rois_per_sweep[sw]:
            ca_trace = ca_df_f[str(roi)]
            # Use the Motor Regressor to remove any motor biased ca activity
            mask_trace = subtract_motor_regressors(ca_trace, binary, ca_sampling_rate)
            # mask_trace2 = mask_ca_trace_by_motor_events(ca_trace, binary, ca_sampling_rate, window=window_sec)
            results.append(mask_trace)
    ca_df_f_no_motor = pd.DataFrame(results).T

    # Store to HDD
    ca_df_f_no_motor.to_csv(f'{base_dir}/data/df_f_data_no_motor.csv', index=False)

    print('==== STORED MOTOR CORRECTED DELTA F OVER F TRACES TO HDD ====')


if __name__ == '__main__':
    main()

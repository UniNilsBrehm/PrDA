import os


class Config:
    # Base directory where all project files are located
    BASE_DIR = os.path.expanduser('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging')
    # BASE_DIR = os.path.expanduser('D:/WorkingData/Initial_Data_PrDA_somas')
    # Define specific directories
    stimulus_protocols_dir: str = os.path.join(BASE_DIR, 'protocols')

    # Define specific files
    # Ventral Root
    vr_events_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_ventral_root_events.h5')
    vr_spontaneous_events_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_ventral_root_spontaneous_events.h5')
    vr_aligned_recordings_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_aligned_vr_recordings.h5')
    vr_binaries_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_ventral_root_binaries.csv')
    vr_all_binaries_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_ventral_root_all_binaries.h5')

    # Ca Imaging
    ca_labels_file: str = os.path.join(BASE_DIR, 'data', 'df_f_data_labels.csv')
    ca_df_f_file: str = os.path.join(BASE_DIR, 'data', 'df_f_data.csv')
    ca_sampling_rate_file: str = os.path.join(BASE_DIR, 'meta_data', 'sampling_rate.csv')
    ca_df_f_sp_motor_file: str = os.path.join(BASE_DIR, 'data', 'df_f_data_spontaneous_motor.csv')
    ca_df_f_no_motor_file: str = os.path.join(BASE_DIR, 'data', 'df_f_data_no_motor.csv')

    # Plotting Data
    ca_responses_spontaneous_motor: str = os.path.join(BASE_DIR, 'figures', 'data',  'ca_responses_spontaneous_motor.csv')
    ca_responses_spontaneous_motor_baseline_corrected: str = os.path.join(BASE_DIR, 'figures', 'data',  'ca_responses_spontaneous_motor_baseline_corrected.csv')

    # Stimulus
    stimulus_traces_file: str = os.path.join(BASE_DIR, 'stimulus_traces', 'stimulus_traces.csv')

    # Linear Scoring
    linear_scoring_file: str = os.path.join(BASE_DIR, 'data', 'linear_scoring_results.csv')

    # SETTINGS:
    # Linear Scoring
    linear_scoring_settings = {
        'max_gap_secs': 10,  # Merge Spontaneous Motor Events that are too close (seconds)
        'tau_rise': 3,  # Rise Time Constant for Calcium Impulse Response Function (seconds)
        'tau_decay': 6,  # Decay Time Constant for Calcium Impulse Response Function (seconds)

        'lag_window': {  # Allowed time shift of regressor for cross correlation to the left / right (seconds) on time axis
            'moving_target_01': [10, 5],
            'moving_target_02': [10, 5],
            'grating_appears': [5, 5],
            'grating_disappears': [5, 5],
            'grating_0': [10, 5],
            'grating_180': [10, 5],
            'bright_loom': [5, 5],
            'dark_loom': [5, 5],
            'bright_flash': [5, 5],
            'dark_flash': [5, 5],
            'motor_spontaneous': [5, 5]
        },

        'time_window': {  # Cutout: Time before and after stimulus period for linear regression model
            'moving_target_01': [2, 10],  # window: -2 to 16 s (stim duration 6 s)
            'moving_target_02': [2, 10],
            'grating_appears': [2, 30],  # window: -2 to 31 s (stim duration 1 s)
            'grating_disappears': [2, 30],
            'grating_0': [2, 12],  # window: -2 to 22 s (stim duration 10 s)
            'grating_180': [2, 12],
            'bright_loom': [2, 30],  # window: -2 to 32 s (stim duration 2 s)
            'dark_loom': [2, 30],
            'bright_flash': [2, 30],  # window: -2 to 31 s (stim duration 1 s)
            'dark_flash': [2, 10],
            'motor_spontaneous': [2, 20]  # window: -2 to 20 s around swim onset
        }

        # 'time_window': {  # Cutout: Time before and after stimulus period for linear regression model
        #     'moving_target_01': [2, 5],  # window: -2 to 11 s (stim duration 6 s)
        #     'moving_target_02': [2, 5],
        #     'grating_appears': [2, 10],  # window: -2 to 11 s (stim duration 1 s)
        #     'grating_disappears': [2, 10],
        #     'grating_0': [2, 2],  # window: -2 to 12 s (stim duration 10 s)
        #     'grating_180': [2, 2],
        #     'bright_loom': [2, 10],  # window: -2 to 12 s (stim duration 2 s)
        #     'dark_loom': [2, 10],
        #     'bright_flash': [2, 10],  # window: -2 to 11 s (stim duration 1 s)
        #     'dark_flash': [2, 10],
        #     'motor_spontaneous': [2, 20]  # window: -2 to 15 s around swim onset
        # }
    }

    # Null Distribution (Permutation/Bootstrapping Test)
    permutation_settings = {
        'n_permutations': 100000,
        'time_before': 2,  # Cutout: Time before event starts (seconds)
        'time_after': 20,  # Cutout: Time after event ends (seconds)

        'tau_rise': 3,  # Rise Time Constant for Calcium Impulse Response Function (seconds)
        'tau_decay': 6  # Decay Time Constant for Calcium Impulse Response Function (seconds)
    }
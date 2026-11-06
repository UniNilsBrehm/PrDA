import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figures.plotting_style import PlotStyle, set_plot_defaults
from utils import norm_min_max
from pathlib import Path
from scipy.stats import kstest, wilcoxon


def get_scores_heatmap(scoring_data, values='score'):
    # Remove multiple entries: Motor Spontaneous and Moving Target 01 and 02
    # There are mean values for these already present
    idx = scoring_data['reg'].isin(['motor_spontaneous_MEAN', 'motor_spontaneous', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor', 'moving_target_02_motor'])
    scoring_data_filtered = scoring_data[~idx].reset_index(drop=True)

    # Pivot data frame to get a long table format for the Scores
    df = scoring_data_filtered.pivot(index=['roi', 'sw'], columns='reg', values=values).reset_index()

    # Rename the columns to remove the _MEAN name
    df = df.rename(columns={
        'moving_target_MEAN': 'moving_target',
        # 'motor_spontaneous_MEAN': 'motor_spontaneous',
        'moving_target_motor_MEAN': 'moving_target_motor'
    })
    return df


def cluster_stimulus_responses():
    file_dir = r'D:\WorkingData\PrTecDA_Data\PrDA_somas_Ca_imaging\data\stimulus_type_score_classification.csv'
    data = pd.read_csv(file_dir, index_col=0)

    # 0: no response
    # 1: mixed (or motor)
    # 2: visual
    classified = (data >= 0.5).astype(int) + 1
    classified.loc['motor'] -= 1
    classified[data == 2] = 0

    filtered = classified.drop(index=['motor', 'dark_loom', 'grating_0'])

    import seaborn as sns
    sns.clustermap(
        filtered,
        metric='hamming',
        method='average',
        cmap='viridis'  # or your custom cmap
    )
    plt.show()


def plot_histograms2(ds, ds_swims, ds_p):
    set_plot_defaults()
    style = PlotStyle()

    # --- plotting ---
    fig_width_cm = 30
    fig_height_cm = 10
    fig_width = float(fig_width_cm / 2.54)
    fig_height = float(fig_height_cm / 2.54)

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    y_lim = 100

    # DS Swims
    panel = 0
    axes[panel].hist(ds_swims, bins=8, density=False, edgecolor='black', facecolor='tab:gray', alpha=1)
    axes[panel].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[panel].set_title("DS based on swim responses \n per sweep")
    axes[panel].set_xlim(-1.1, 1.1)
    axes[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[panel].set_xlabel("Direction selectivity index")
    axes[panel].set_ylim(0, y_lim)
    axes[panel].set_yticks([0, 25, 50, 75, 100])
    axes[panel].set_ylabel("Count")

    # DS Ca Responses
    panel = 1
    axes[panel].hist(ds, bins=15, density=False, edgecolor='black', facecolor='tab:green', alpha=1)
    axes[panel].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[panel].set_title("DS based on stimulus reg. scores \n per neuron")
    axes[panel].set_xlim(-1.1, 1.1)
    axes[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[panel].set_xlabel("Direction selectivity index")
    axes[panel].set_ylim(0, y_lim)
    axes[panel].set_yticks([0, 25, 50, 75, 100])
    axes[panel].set_ylabel("Count")

    # DS Penalty
    panel = 2
    axes[panel].hist(ds_p, bins=15, density=False, edgecolor='black', facecolor='tab:olive', alpha=1)
    axes[panel].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[panel].set_title("Weighted DS based on stimulus reg. scores \n per neuron")
    axes[panel].set_xlim(-1.1, 1.1)
    axes[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[panel].set_xlabel("Direction selectivity index")
    axes[panel].set_ylim(0, y_lim)
    axes[panel].set_yticks([0, 25, 50, 75, 100])
    axes[panel].text(
        -0.90, 90,
        r'$F^*=\frac{F}{1+N_F}$' '\n' 
        r'$B^*=\frac{B}{1+N_B}$' '\n'
        r'$\mathrm{DS}=\frac{F^*-B^*}{F^*+B^*}$',
        fontsize=10,
        va='top'
    )

    plt.show()


def plot_histograms(ds, ds_motor, fw, fw_motor, bw, bw_motor, bins=40):
    set_plot_defaults()
    style = PlotStyle()

    fw_norm = norm_min_max(fw)
    fw_motor_norm = norm_min_max(fw_motor)
    bw_norm = norm_min_max(bw)
    bw_motor_norm = norm_min_max(bw_motor)

    # --- plotting ---
    fig_width_cm = 30
    fig_height_cm = 5
    fig_width = float(fig_width_cm / 2.54)
    fig_height = float(fig_height_cm / 2.54)

    fig, axes = plt.subplots(1, 5, figsize=(fig_width, fig_height))
    y_lim = 100

    # FW: Stim vs Motor Scores
    panel = 0
    axes[panel].scatter(fw_motor_norm, fw_norm, **style.scCorr)
    axes[panel].set_xlabel('FW Motor Score')
    axes[panel].set_ylabel('FW Stim Score')
    axes[panel].set_xlim(-0.1, 1.1)
    axes[panel].set_ylim(-0.1, 1.1)

    # BW: Stim vs Motor Scores
    panel = 1
    axes[panel].scatter(bw_motor_norm, bw_norm, **style.scCorr)
    axes[panel].set_xlabel('BW Motor Score')
    axes[panel].set_ylabel('BW Stim Score')
    axes[panel].set_xlim(-0.1, 1.1)
    axes[panel].set_ylim(-0.1, 1.1)

    # DS
    panel = 2
    axes[panel].hist(ds, bins=bins, density=False, edgecolor='black', facecolor='tab:green', alpha=1)
    axes[panel].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[panel].set_title("DS=(fw-bw)/(fw+bw)")
    axes[panel].set_xlim(-1.1, 1.1)
    axes[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[panel].set_xlabel("Direction selectivity index")
    axes[panel].set_ylim(0, y_lim)
    axes[panel].set_yticks([0, 25, 50, 75, 100])
    axes[panel].set_ylabel("Count")

    # DS Motor
    panel = 3
    axes[panel].hist(ds_motor, bins=bins, density=False, edgecolor='black', facecolor='tab:blue', alpha=1)
    axes[panel].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[panel].set_title("DS motor")
    axes[panel].set_xlim(-1.1, 1.1)
    axes[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[panel].set_xlabel("Direction selectivity index")
    axes[panel].set_ylim(0, y_lim)
    axes[panel].set_yticks([0, 25, 50, 75, 100])

    # DS Motor vs DS Stimulus
    panel = 4
    axes[panel].scatter(ds_motor, ds, **style.scCorr)
    axes[panel].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[panel].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[panel].set_xlabel('DS Motor')
    axes[panel].set_ylabel('DS Stimulus')
    axes[panel].set_xlim(-1.1, 1.1)
    axes[panel].set_ylim(-1.1, 1.1)
    axes[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[panel].set_yticks([-1, -0.5, 0, 0.5, 1])

    for ax in axes:
        ax.spines['right'].set_visible(False)  # Hide right spine
        ax.spines['top'].set_visible(False)  # Hide top spine (y-axis line)

    plt.show()


def remove_close_to_zero(data1, data2, tolerance=0.05):
    mask = ~(np.isclose(data1, 0, atol=tolerance) & np.isclose(data2, 0, atol=tolerance))
    return mask


def do_tests_on_ds_data(direction_selectivity):
    # direction_selectivity = pd.read_csv(f'final_figures/data/fig6/direction_selectivity.csv', index_col=0)
    ds = direction_selectivity['direction_selectivity_stimulus']
    ds_swim = direction_selectivity['direction_selectivity_swims']
    ds_corrected = direction_selectivity['direction_selectivity_corrected']

    results = pd.DataFrame()
    results['ds_stimulus'] = ds
    results['ds_swim'] = ds_swim
    results['ds_corrected'] = ds_corrected

    # Run Kolmogorov-Smirnov test against uniform distribution
    stat_ds, p_ds = kstest(ds.to_numpy(), 'uniform', args=(-1, 2))
    stat_swim, p_swim = kstest(ds_swim.to_numpy(), 'uniform', args=(-1, 2))
    stat_corrected, p_corrected = kstest(ds_corrected.to_numpy(), 'uniform', args=(-1, 2))

    ks_results = pd.DataFrame()
    ks_results['ds'] = ['stimulus', 'swim', 'corrected']
    ks_results['stat'] = [stat_ds, stat_swim, stat_corrected]
    ks_results['p'] = [p_ds, p_swim, p_corrected]

    # Wilcoxon signed-rank test against median = 0
    stat_ds_w, p_ds_w = wilcoxon(ds, alternative='two-sided')
    stat_swim_w, p_swim_w = wilcoxon(ds_swim, alternative='two-sided')
    stat_corrected_w, p_corrected_w = wilcoxon(ds_corrected, alternative='two-sided')

    wilcoxon_results = pd.DataFrame({
        'ds': ['stimulus', 'swim', 'corrected'],
        'stat': [stat_ds_w, stat_swim_w, stat_corrected_w],
        'p': [p_ds_w, p_swim_w, p_corrected_w]
    })

    # Store to HDD
    results.to_csv(f'final_figures/data/fig6/ds_results.csv')
    ks_results.to_csv(f'final_figures/data/fig6/ds_ks_test.csv')
    wilcoxon_results.to_csv('final_figures/data/fig6/ds_wilcoxon_test.csv')

    print('DS data stored to HDD')


def main():
    scores_org = pd.read_csv(r'D:\WorkingData\PrTecDA_Data\PrDA_somas_Ca_imaging\data\linear_scoring_results.csv')
    scores = get_scores_heatmap(scores_org, values='score')

    scores_fw = scores['grating_0']
    scores_fw_motor = scores['grating_0_motor']
    scores_bw = scores['grating_180']
    scores_bw_motor = scores['grating_180_motor']

    # --- Remove negative scores ---
    fw = scores_fw.clip(lower=0)
    bw = scores_bw.clip(lower=0)
    fw_motor = scores_fw_motor.clip(lower=0)
    bw_motor = scores_bw_motor.clip(lower=0)

    # tolerance = 0.01
    # mask1 = remove_close_to_zero(fw_motor, fw_motor, tolerance=tolerance)
    # mask2 = remove_close_to_zero(bw_motor, bw_motor, tolerance=tolerance)
    # mask = mask1 * mask2
    #
    # fw = fw[mask]
    # bw = bw[mask]
    # fw_motor = fw_motor[mask]
    # bw_motor = bw_motor[mask]

    # --- DS ---
    den = fw + bw
    ds = ((fw - bw) / den).where(den > 0, 0)

    den_motor = fw_motor + bw_motor
    ds_motor = ((fw_motor - bw_motor) / den_motor).where(den_motor > 0, 0)

    # mask = den_motor > 0
    # fw = fw[mask]
    # fw_motor = fw_motor[mask]
    # bw = bw[mask]
    # bw_motor = bw_motor[mask]
    # ds = ds[mask]
    # ds_motor = ds_motor[mask]

    # store data for plotting
    base_dir = Path(r'C:\UniFreiburg\Code\PrDA\final_figures\data\fig6')
    ds_data = pd.DataFrame()
    ds_data['forward_grating_stimulus_score'] = fw
    ds_data['forward_grating_motor_score'] = fw_motor
    ds_data['backward_grating_stimulus_score'] = bw
    ds_data['backward_grating_motor_score'] = bw_motor
    ds_data['direction_selectivity_stimulus'] = ds
    ds_data['direction_selectivity_motor'] = ds_motor

    # Use swim number as a penalty for DS
    data_dir = f'final_figures/data/supp_fig4'
    data_labels = pd.read_csv('final_figures/data/fig6/df_f_data_labels.csv')

    ds_data['sw'] = data_labels.loc[1].to_numpy()

    swim_data = pd.read_csv(
        f'{data_dir}/swim_bouts_per_stimulus.csv',
        index_col=0
    )

    swim_data['sw'] = data_labels.loc[1].unique()

    # Merge both data frames
    df = ds_data.merge(swim_data, on="sw", how="left")

    fw_swims = df['grating_0']
    bw_swims = df['grating_180']
    a = 1
    fw_p = fw / (1 + a * fw_swims)
    bw_p = bw / (1 + a * bw_swims)
    ds_p = (fw_p - bw_p) / (fw_p + bw_p)

    # DS based on swim responses
    ds_swims = (fw_swims - bw_swims) / (fw_swims + bw_swims)
    # plot_histograms2(ds, ds_swims, ds_p)

    # Add to data set
    ds_data['direction_selectivity_swims'] = ds_swims
    ds_data['direction_selectivity_corrected'] = ds_p
    ds_data.to_csv(base_dir / 'direction_selectivity.csv')
    print('Stored DS data to HDD')

    # Now to statistical tests and save all results to csv file directly to the corresponding figure
    # final_figures -> data/fig6
    do_tests_on_ds_data(ds_data)


if __name__ == '__main__':
    main()

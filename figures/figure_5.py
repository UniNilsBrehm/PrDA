"""
Assemble and export Figure 5 for the manuscript:
A dopaminergic visuomotor pathway to the zebrafish optic tectum
Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann
iScience, Cell Press 2026

Figure 5: PrDA neurons combine motor and visual inputs

"""

# Imports
import numpy as np
import pandas as pd
from plotting_style import PlotStyle, set_plot_defaults
from plotting_utils import hide_axis_spines
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import mannwhitneyu
from utils import load_hdf5_as_data_frame


# Helper Functions
def load_scoring_data():
    null_scores_file = f'data/fig5/scores_null_distribution_10k_permutations.h5'
    null_scores = load_hdf5_as_data_frame(null_scores_file).iloc[:, 0].to_numpy()
    linear_scores = pd.read_csv(f'data/fig5/linear_scoring_results.csv')

    # Remove negative scores (meaningless)
    idx = linear_scores['score'] < 0
    linear_scores.loc[idx, 'score'] = 0

    s1 = 'bright_loom'
    s2 = 'dark_loom'
    s1_motor = 'bright_loom_motor'
    s2_motor = 'dark_loom_motor'

    scores = {
        'bl_scores': linear_scores[linear_scores['reg'] == s1].reset_index(drop=True),
        'bl_motor_scores': linear_scores[linear_scores['reg'] == s1_motor].reset_index(drop=True),
        'dl_scores': linear_scores[linear_scores['reg'] == s2].reset_index(drop=True),
        'dl_motor_scores': linear_scores[linear_scores['reg'] == s2_motor].reset_index(drop=True)
    }

    return scores, null_scores, linear_scores


def get_score_data():
    linear_scoring_data = pd.read_csv(
        f'data/fig4/linear_scoring_results.csv'
    )
    neuron_data = prepare_visual_neuron_counts(linear_scoring_data)
    neuron_df = neuron_data.copy()

    scale_factor = 0.131  # for converting pixels to um
    x_midline = (neuron_df["y"].max() + neuron_df["y"].min()) / 2
    y_midline = (neuron_df["x"].max() + neuron_df["x"].min()) / 2
    neuron_df["x_plot"] = (neuron_df["y"] - x_midline) * scale_factor
    neuron_df["y_plot"] = (neuron_df["x"] - y_midline) * scale_factor

    # Separate left and right side
    neuron_df["side"] = np.where(neuron_df["x_plot"] < 0, "left", "right")

    # Mann–Whitney U test: Left vs. Right
    left = neuron_df[neuron_df['side'] == 'left']['n_visual_stimuli']
    right = neuron_df[neuron_df['side'] == 'right']['n_visual_stimuli']
    stat, p = mannwhitneyu(left, right, alternative='two-sided')

    return neuron_df, p


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


def get_spontaneous_scores(scores_org):
    # Remove not needed entries
    idx = scores_org['reg'].isin(['motor_spontaneous_MEAN', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor', 'moving_target_02_motor'])
    df_scores = scores_org[~idx].reset_index(drop=True)

    # Get spontaneous swim events data
    df_sp = df_scores[df_scores['reg'] == 'motor_spontaneous'].copy().reset_index(drop=True)
    df_sp["event_idx"] = df_sp.groupby("roi").cumcount()

    df_sub_sp = df_sp[['roi', 'sw', 'fish', 'score_norm99', 'score', 'r2', 'coeff', 'lag_sec', 'threshold', 'threshold_r2', 'threshold_slope', 'significant', 'event_idx']].copy()

    df_per_roi = (
        df_sub_sp.groupby(["roi"])["significant"]
        .agg(
            n_significant="sum",  # number of True
            total_events="count"  # total rows
        )
        .assign(fraction=lambda x: x["n_significant"] / x["total_events"])
        .reset_index()
    )
    return df_sub_sp, df_per_roi


def prepare_visual_neuron_counts(df):
    """
    Prepare one row per neuron with number of purely visual stimulus types.
    """
    df = df[
        (~df["reg"].isin(["motor_spontaneous"])) &
        (~df["reg"].str.contains("_MEAN"))
    ].copy()

    df_stim = df[~df["reg"].str.endswith("_motor")].copy()
    df_motor = df[df["reg"].str.endswith("_motor")].copy()

    df_stim["reg_base"] = df_stim["reg"]
    df_motor["reg_base"] = df_motor["reg"].str.replace("_motor", "", regex=False)
    df_motor = df_motor.rename(columns={"significant": "significant_motor"})

    merged = pd.merge(
        df_stim,
        df_motor[["roi", "reg_base", "significant_motor"]],
        on=["roi", "reg_base"],
        how="left"
    )

    merged["significant_motor"] = merged["significant_motor"].fillna(False)

    merged["visual_only"] = (
        (merged["significant"] == True) &
        (merged["significant_motor"] == False)
    )

    all_neurons = (
        merged.groupby("roi", as_index=False)
        .agg({
            "x_position": "first",
            "y_position": "first"
        })
        .rename(columns={
            "x_position": "x",
            "y_position": "y"
        })
    )

    visual_counts = (
        merged[merged["visual_only"]]
        .groupby("roi", as_index=False)
        .agg({"reg_base": "nunique"})
        .rename(columns={"reg_base": "n_visual_stimuli"})
    )

    neuron_df = pd.merge(all_neurons, visual_counts, on="roi", how="left")
    neuron_df["n_visual_stimuli"] = neuron_df["n_visual_stimuli"].fillna(0).astype(int)

    return neuron_df


def neuron_classification(scores_org):
    # This is the classification of neurons onto M+/V+ etc. based on all stimulus types and motor events averaged.
    # This goes into Figure 5 Pie Chart.
    stimulus_list = [
        'moving_target_01',
        'moving_target_02',
        'grating_appears',
        'grating_0',
        'grating_180',
        'grating_disappears',
        'bright_loom',
        'dark_loom',
        'bright_flash',
        'dark_flash',
    ]

    neuron_classes = list()

    for roi in scores_org['roi'].unique():
        # motor_only = 0
        both = 0
        enough_motor = 1
        only_vis_driven = 0

        d = scores_org[scores_org['roi'] == roi]
        idx_motor = d['reg'] == 'motor_spontaneous'
        motors = d[idx_motor]['significant']

        if motors.shape[0] >= 3:  # at least 3 motor events
            if motors.sum() >= 1:  # one has to be sign.
                motor_only = 1
            else:
                motor_only = 0
        else:
            enough_motor = 0  # not enough data to decide
            motor_only = 2

        vis_only = 0
        for reg in stimulus_list:
            s_vis = d[d['reg'] == reg]['significant'].item()
            s_motor = d[d['reg'] == f"{reg}_motor"]['significant'].item()
            if s_vis and not s_motor:
                vis_only = 1  # Now we hit at least one stimulus with this condition, the rest does not matter

        if vis_only == 1 and motor_only == 0:
            only_vis_driven = 1

        row = [roi, vis_only, motor_only, enough_motor, only_vis_driven]
        neuron_classes.append(row)

    neuron_classes_df = pd.DataFrame(neuron_classes, columns=['roi', 'vis', 'motor', 'enough_motor', 'only_vis_driven'])
    neuron_classes_df['vis'].value_counts()
    neuron_classes_df['motor'].value_counts()
    neuron_classes_df['enough_motor'].value_counts()

    motor_driven_v = ((neuron_classes_df['vis'] == 1) & (neuron_classes_df['motor'] == 1)).sum()
    motor_driven_nv = ((neuron_classes_df['vis'] == 0) & (neuron_classes_df['motor'] == 1)).sum()
    motor_unknown = ((neuron_classes_df['vis'] == 1) & (neuron_classes_df['motor'] == 2)).sum()
    motor_not = ((neuron_classes_df['vis'] == 1) & (neuron_classes_df['motor'] == 0)).sum()

    results = [motor_driven_v, motor_driven_nv, motor_unknown, motor_not]
    results.extend(
        [
            neuron_classes_df['vis'].value_counts()[1],
            neuron_classes_df['motor'].value_counts()[1],
            neuron_classes_df['motor'].value_counts()[2],
            neuron_classes_df['motor'].value_counts()[0],
            neuron_classes_df['only_vis_driven'].value_counts()[1]
        ]
    )

    summary_df = pd.DataFrame(
        [results],
        columns=[
            'motor_driven_v',
            'motor_driven_nv',
            'motor_unknown',
            'motor_not',
            'visual_driven',
            'motor_driven',
            'motor_unknown',
            'not_motor_driven',
            'only_vis_driven',
        ])

    # summary_df.loc["fraction"] = summary_df.loc[0] / 360
    summary_df = summary_df.rename(index={0: "counts"})

    return summary_df


def run_classification():
    scores_org = pd.read_csv(f'data/fig4/linear_scoring_results.csv')
    # Classify neurons
    summary = neuron_classification(scores_org)
    summary.to_csv(f'data/fig5/classification_summary.csv')
    print('==== Stored Classification Summary to HDD')


# Plotting functions
def plot_si(ax, df_si, si_bootstrap):
    style = PlotStyle()
    # ---- Setup ----
    metric_map = {
        "Vis vs Spont": "vis_spont_index",
        "Mixed vs Spont": "mixed_spont_index",
        "Mixed vs Vis": "mixed_vis_index",
        "Synergy Index": "synergy_index",
        # "Synergy Ratio": "synergy"
    }
    x_labels = ['vis motor', 'both motor', 'both vis', 'synergy']
    si_bootstrap = si_bootstrap[si_bootstrap["name"].isin(metric_map.keys())]

    names = list(metric_map.keys())
    M = len(names)

    fish_ids = sorted(df_si["fish"].unique())
    F = len(fish_ids)
    gap = 2  # space between metrics

    # ---- Bootstrap estimates + CIs ----
    means = si_bootstrap["Estimate"].values
    lowCI = si_bootstrap["CI_low"].values
    highCI = si_bootstrap["CI_high"].values
    yerr = np.vstack([means - lowCI, highCI - means])

    # center of each metric group
    metric_centers = np.array([m * (F + gap) + F / 2 for m in range(M)])

    ax.errorbar(
        metric_centers,
        means,
        yerr=yerr,
        color='black',
        ecolor='black',
        label="Bootstrap estimate",
        **style.error_bar
    )

    # ---- Add significance stars ----
    stars = ['', '***', '***', '']
    y_pos = 0.5
    for i in range(M):
        star = stars[i]
        if star:
            x = metric_centers[i]
            # y = highCI[i]  # top of CI bar
            y = y_pos
            # vertical offset for visibility (tune if needed)
            offset = 0.5

            ax.text(
                x, y + offset, star,
                ha='center', va='bottom',
                fontsize=12, color='black', weight='bold'
            )

    # ---- Raw point clouds grouped by fish ----
    colors = plt.cm.tab20(np.linspace(0, 1, F))

    for fi, fish in enumerate(fish_ids):
        df_f = df_si[df_si["fish"] == fish]

        for mi, metric in enumerate(names):
            x_base = mi * (F + gap) + fi

            # jitter for cloud
            jitter = np.random.normal(scale=0.15, size=len(df_f))
            yvals = df_f[metric_map[metric]].values

            ax.scatter(
                x_base + jitter,
                yvals,
                color=colors[fi],
                alpha=0.5,
                s=5,
                label=f"Fish {fish}" if mi == 0 else None  # legend once
            )

    # ---- Formatting ----
    ax.axhline(0, color='black', linestyle=':', linewidth=1.5)
    ax.axhline(0.33, color='gray', linestyle=':', linewidth=1)
    ax.axhline(-0.33, color='gray', linestyle=':', linewidth=1)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax.axhline(-0.5, color='gray', linestyle=':', linewidth=1)
    ax.text(30.5, 0+0.01, '1x', ha='center', va='bottom', fontsize=6, color='black', weight='bold')
    ax.text(30.5, 0.33+0.01, '2x', ha='center', va='bottom', fontsize=6, color='black', weight='bold')
    ax.text(30.5, 0.5+0.01, '3x', ha='center', va='bottom', fontsize=6, color='black', weight='bold')
    ax.text(30.5, -0.33-0.01, '1/2x', ha='center', va='bottom', fontsize=6, color='black', weight='bold')
    ax.text(30.5, -0.5-0.01, '1/3x', ha='center', va='bottom', fontsize=6, color='black', weight='bold')

    # x-axis ticks in center of groups
    ax.set_xticks(metric_centers)
    # ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xticklabels(x_labels, rotation=0, ha='center')

    ax.set_ylim((-1, 1))
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("SI value")
    # ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))


def plot_class_heatmap(ax, scores_org, thresholds):
    style = PlotStyle()
    scores_df = pd.read_csv(f'data/fig4/linear_scoring_results.csv')
    df_sp, df_roi1 = get_spontaneous_scores(scores_df)

    # Get Neurons (rois) that saw at least 3 spontaneous motor events
    idx = df_roi1['total_events'] < 3
    df_roi = df_roi1[~idx].copy()  # remove those from the total number of rois

    col_order = [
        'dark_loom', 'dark_loom_motor',
        'grating_0', 'grating_0_motor',
        'grating_180', 'grating_180_motor',
        'grating_appears', 'grating_appears_motor',
        'grating_disappears', 'grating_disappears_motor',
        'bright_loom', 'bright_loom_motor',
        'dark_flash', 'dark_flash_motor',
        'moving_target', 'moving_target_motor',
        'bright_flash', 'bright_flash_motor',
    ]
    col_labels = [
        'DL',
        'GF',
        'GB',
        'GA',
        'GD',
        'BL',
        'DF',
        'MT',
        'BF'
    ]

    # Compute Normalized Difference (Selectivity Index) of Scores
    # D = (stim - motor) / (stim + motor)
    # Bootstrapped thresholds per neuron
    scores = scores_org.drop(columns=['roi', 'sw']).reset_index(drop=True)
    scores.columns.name = None

    th = thresholds.iloc[:, 0].reset_index(drop=True)
    scores_metric = pd.DataFrame()
    for i in np.arange(0, len(col_order) - 1, 2):  # loop over stimuli
        s = col_order[i]
        m = col_order[i + 1]
        s_scores = scores[s]
        m_scores = scores[m]

        # Set all scores smaller than zero to zero (minus scores a useless)
        s_scores[s_scores < 0] = 0
        m_scores[m_scores < 0] = 0

        # Get no-responses
        idx_no = s_scores <= th

        # D = (stim - motor) / (stim + motor)
        s_pair = (s_scores - m_scores) / (s_scores + m_scores)

        # Set scores below threshold to: no response (set to 2)
        s_pair[idx_no] = 2
        scores_metric[s] = s_pair

    # Now check if neurons are driven by spontaneous motor events
    result = scores_metric.copy()
    result['roi'] = scores_org['roi']

    # Find rois that are driven by spontaneous motor events
    # definition: At least 1 sign. motor event (from at least 3 in total)
    # idx = df_roi['fraction'] >= 0.1
    idx = df_roi['n_significant'] >= 1
    rois_motor = df_roi['roi'][idx]
    result['motor'] = result["roi"].isin(df_roi['roi'].values).astype(int) + result["roi"].isin(rois_motor.values).astype(int)

    # motor=0: not enough data to decide
    # motor=1: not motor driven
    # motor=2: motor driven

    result.drop(columns=['roi'], inplace=True)
    result['motor'] = result['motor'].map({0: 2, 2: 0, 1: 1})
    plot_data = result.T  # transpose: stim → rows, neurons → cols
    # ---------------------------------------------------------------------------------------------------------------
    # (0=mixed, 1=stimulus, 2=no response)
    # Boundaries so each integer maps to its own color
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, style.cmap_classes.N)

    # Separate data
    plot_data_main = plot_data.iloc[:-1, :]  # all stim rows
    plot_data_motor = plot_data.iloc[-1:, :]  # M row

    # ---------------------------------------------------------------------------------------------------------------
    # Plot stim/motor classes for stimulus types
    # Normal colormap for stim rows
    ax.pcolormesh(
        np.arange(plot_data_main.shape[1] + 1),
        np.arange(plot_data_main.shape[0] + 1),
        plot_data_main,
        cmap=style.cmap_classes,
        norm=norm,
        rasterized=True
    )

    # Plot extra spontaneous motor row on top of the heatmap
    # Custom colormap for motor row
    motor_cmap = ListedColormap(["tab:blue", "tab:red", "white"])  # motor, vis, unknown
    bounds_motor = [-0.5, 0.5, 1.5, 2.5]
    norm_motor = BoundaryNorm(bounds_motor, motor_cmap.N)
    ax.pcolormesh(
        np.arange(plot_data_motor.shape[1] + 1),
        np.arange(plot_data_main.shape[0], plot_data_main.shape[0] + plot_data_motor.shape[0] + 1),
        plot_data_motor,
        cmap=motor_cmap,
        norm=norm_motor,
        rasterized=True
    )

    # Y axis: stimulus labels
    col_labels.append('M')
    ax.set_yticks(np.arange(len(col_labels)) + 0.5)
    ax.set_yticklabels(col_labels, fontsize=6)

    # X axis: every 100th neuron
    step = 100
    xticks = np.arange(0, plot_data.shape[1], step)
    ax.set_xticks(xticks)


def plot_scores_heatmap(ax, scores, v_min, v_max):
    col_order = [
        'dark_loom', 'dark_loom_motor',
        'grating_0', 'grating_0_motor',
        'grating_180', 'grating_180_motor',
        'grating_appears', 'grating_appears_motor',
        'grating_disappears', 'grating_disappears_motor',
        'bright_loom', 'bright_loom_motor',
        'dark_flash', 'dark_flash_motor',
        'moving_target', 'moving_target_motor',
        'bright_flash', 'bright_flash_motor',
        # 'motor_spontaneous'
    ]

    col_labels = [
        'vDL', 'mDL',
        'vGF', 'mGF',
        'vGB', 'mGB',
        'vGA', 'mGA',
        'vGD', 'mGD',
        'vBL', 'mBL',
        'vDF', 'mDF',
        'vMT', 'mMT',
        'vBF', 'mBF',
        # 'sp. motor'
    ]

    # Instead of reordering columns → reorder rows
    scores_sorted = scores[col_order].T  # transpose: stim → rows, neurons → cols

    ax.pcolormesh(
        np.arange(scores_sorted.shape[1] + 1),  # neurons on x-axis
        np.arange(scores_sorted.shape[0] + 1),  # stimuli on y-axis
        scores_sorted.values,
        cmap='afmhot',
        vmin=v_min,
        vmax=v_max,
        rasterized=True
    )

    # Horizontal lines separating stimulus groups
    for y in range(2, scores_sorted.shape[0], 2):
        ax.axhline(y, color="white", linewidth=1.0, linestyle='-', alpha=1.0)

    # Y axis: stimulus labels
    ax.set_yticks(np.arange(scores_sorted.shape[0]) + 0.5)
    yticklabels = ax.set_yticklabels(col_labels, fontsize=6)

    # X axis: every 100th neuron
    step = 100
    xticks = np.arange(0, scores_sorted.shape[1], step)
    ax.set_xticks(xticks)

    # alternate colors: even → green, odd → blue
    for i, text in enumerate(yticklabels):
        if i % 2 == 0:
            text.set_color("green")
        else:
            text.set_color("blue")


# Main functions
def create_figure():
    set_plot_defaults()
    style = PlotStyle()
    base_dir = 'data/fig5'

    # Load setup image
    cartoon_results = plt.imread(f'{base_dir}/images/integration_cartoon_results.png')

    # Load data
    df_si = pd.read_csv(f"{base_dir}/score_metrics_per_neuron.csv", index_col=0)
    si_bootstrap = pd.read_csv(f"{base_dir}/selectivity_index_bootstrap.csv")

    # Classify neurons
    summary_df = pd.read_csv(f'{base_dir}/classification_summary.csv', index_col=0)

    # Prepare classification pie plot
    classes = [summary_df['motor_driven_v'].item(), summary_df['motor_driven_nv'].item(), summary_df['motor_unknown'].item() + 1, summary_df['motor_not'].item()]
    colors = style.pie_colors
    annotations = [
        "M+/V+ (276)",
        "M+/V- (23)",
        "M?/V+ (57)",
        "M-/V+ (4)"
    ]

    # Load linear scoring results
    scores, null_scores, linear_scores = load_scoring_data()

    # Compute heatmap data
    scores_heatmap = get_scores_heatmap(linear_scores, values='score')
    scores_heatmap_norm = get_scores_heatmap(linear_scores, values='score_norm99')

    # Get Thresholds
    thresholds_file = f'{base_dir}/linear_scoring_thresholds.csv'
    thresholds = pd.read_csv(thresholds_file, index_col=0)

    # ==================================================================================================================
    # Figure
    # A4 page: The figure will be on a DinA4 page in the final pdf file
    a4_w_cm, a4_h_cm = 21.0, 29.7
    fig_width = a4_w_cm / 2.54
    fig_height = a4_h_cm / 2.54

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Center the 16x20 cm content block on the A4 page
    content_w_cm, content_h_cm = 16, 12
    left = ((a4_w_cm - content_w_cm) / 2) / a4_w_cm
    right = 1 - left
    bottom = ((a4_h_cm - content_h_cm) / 2) / a4_h_cm
    top = 1 - bottom

    gs_left = left + 0.1 * (right - left)
    gs_right = left + 0.95 * (right - left)
    gs_bottom = bottom + 0.06 * (top - bottom)
    gs_top = bottom + 0.95 * (top - bottom)

    grid = matplotlib.gridspec.GridSpec(
        nrows=int(content_h_cm * 10), ncols=int(content_w_cm * 10),
        left=gs_left, right=gs_right, bottom=gs_bottom, top=gs_top,
        wspace=0, hspace=0,
    )

    # ------------------------------------------------------------------------------------------------------------------
    axs = dict()
    # Define axes
    # Score heatmap
    axs['scores_heatmap'] = plt.subplot(grid[0:40, 0:40])
    axs['scores_heatmap_cb'] = plt.subplot(grid[5:35, 42:44])

    # Classes heatmap
    axs['classes_heatmap'] = plt.subplot(grid[0:40, 70:150])
    axs['classes_heatmap_cb'] = plt.subplot(grid[5:35, 152:154])

    # Classification Pi
    axs['classification'] = plt.subplot(grid[65:85, 0:20])

    # SI
    axs['si'] = plt.subplot(grid[65:105, 40:120])

    # Summary Cartoon
    axs['cartoon'] = plt.subplot(grid[70:100, 130:160])  # 30 x 40

    # ==================================================================================================================
    # A) Regressor Scores Heatmap
    v_min = 0
    v_max = 1
    scores_heatmap_norm_plotting = scores_heatmap_norm.drop(columns=['roi', 'sw']).reset_index(drop=True)
    scores_heatmap_norm_plotting.columns.name = None
    plot_scores_heatmap(axs['scores_heatmap'], scores_heatmap_norm_plotting, v_min, v_max)
    axs['scores_heatmap'].set_ylabel('Regressors')
    axs['scores_heatmap'].set_xlabel('Neuron #')
    axs['scores_heatmap'].set_xticks([0, 100, 200, 300])

    # scores heatmap colorbar
    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    cb = matplotlib.colorbar.ColorbarBase(axs['scores_heatmap_cb'], norm=norm, cmap=style.heatmap_cmap,
                                          orientation='vertical')
    cb.ax.tick_params(labelsize=style.cbar_ticks_size)
    cb.set_label('score', rotation=270, labelpad=10, size=style.cbar_label_size)
    cb.set_ticks([0.0, 0.5, 1.0])

    # ==================================================================================================================
    # B) Classification Heatmap
    plot_class_heatmap(axs['classes_heatmap'], scores_heatmap, thresholds)
    axs['classes_heatmap'].set_ylabel('Stimulus type')
    axs['classes_heatmap'].set_xlabel('Neuron #')
    axs['classes_heatmap'].set_xticks([0, 100, 200, 300])

    # ------------------------------------------------------------------------------------------------------------------
    # Class heatmap colorbar
    cb = matplotlib.colorbar.ColorbarBase(axs['classes_heatmap_cb'], cmap=style.cmap_classes, orientation='vertical')
    cb.ax.tick_params(labelsize=style.cbar_ticks_size)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['mixed', 'visual', 'no'])

    # ==================================================================================================================
    # C) Classification Pie Plot
    wedges, texts = axs['classification'].pie(
        classes,
        colors=colors,
        labels=None,  # no labels on the slices
        startangle=90,
        # autopct="%1.0f%%"
    )

    axs['classification'].legend(
        wedges,
        annotations,
        loc="center",  # choose location
        bbox_to_anchor=(0.6, -0.5),  # place legend outside the plot
        fontsize=8,  # smaller text
        title_fontsize=9,  # optional
        frameon=False,  # cleaner look (optional)
        ncol=1,       # makes the legend two columns
        handlelength=1,   # shorten color box
        handleheight=1    # reduce height
    )

    # ==================================================================================================================
    # D) Selectivity Indices Plot
    plot_si(axs['si'], df_si, si_bootstrap)
    text_h = -0.30
    axs['si'].text(
        0.14, text_h,
        r'$\frac{\mu_{\mathrm{vis}} - \mu_{\mathrm{motor}}}'
        r'{\mu_{\mathrm{vis}} + \mu_{\mathrm{motor}}}$',
        transform=axs['si'].transAxes,
        **style.txtSI
    )

    axs['si'].text(
        0.38, text_h,
        r'$\frac{\mu_{\mathrm{both}} - \mu_{\mathrm{motor}}}'
        r'{\mu_{\mathrm{both}} + \mu_{\mathrm{motor}}}$',
        transform=axs['si'].transAxes,
        **style.txtSI
    )

    axs['si'].text(
        0.64, text_h,
        r'$\frac{\mu_{\mathrm{both}} - \mu_{\mathrm{vis}}}'
        r'{\mu_{\mathrm{both}} + \mu_{\mathrm{vis   }}}$',
        transform=axs['si'].transAxes,
        **style.txtSI
    )

    axs['si'].text(
        0.9, text_h,
        r'$\frac{\mu_{\mathrm{both}} - (\mu_{\mathrm{vis}} + \mu_{\mathrm{motor}})}'
        r'{\mu_{\mathrm{both}} + \mu_{\mathrm{vis}} + \mu_{\mathrm{motor}}}$',
        transform=axs['si'].transAxes,
        **style.txtSI
    )

    # ==================================================================================================================
    # E) Results Cartoon
    axs['cartoon'].imshow(cartoon_results, aspect='auto', interpolation=None)

    # ==================================================================================================================
    # Hide axes
    axs['cartoon'].axis('off')
    hide_axis_spines(axs['si'])

    # ==================================================================================================================
    # Add Sub-Figure Caps
    # from plotting_utils import set_caption_labels
    # set_caption_labels(fig)
    # plt.show()
    # exit()

    cap_pos = {
        'A': (np.float64(0.02), np.float64(0.96)),
        'B': (np.float64(0.40), np.float64(0.96)),
        'C': (np.float64(0.02), np.float64(0.48)),
        'D': (np.float64(0.26), np.float64(0.48)),
        'E': (np.float64(0.77), np.float64(0.48)),
    }

    # Redefine position to DinA4 page
    def content_xy(fx, fy):
        return left + fx * (right - left), bottom + fy * (top - bottom)

    for cap, (fx, fy) in cap_pos.items():
        x, y = content_xy(fx, fy)
        fig.text(x, y, cap, **style.cap_style)

    # ==================================================================================================================
    # Store Figure to HDD
    fig_dir = f"figs"
    matplotlib.rcParams["savefig.bbox"] = None
    fig_nr = 5

    # Add figure name to pdf
    fig.text(style.figure_label_pos[0], style.figure_label_pos[1], f'Figure {fig_nr}', **style.figure_label)

    plt.savefig(f'{fig_dir}/05_figure_{fig_nr}.pdf', dpi=600)

    plt.close(fig)
    print(f'\n=== STORED FIGURE {fig_nr} TO HDD ====\n')


def main():
    # Run classification
    run_classification()

    # Create figure
    create_figure()


if __name__ == "__main__":
    main()

"""
Assemble and export Figure 6 for the manuscript:
A dopaminergic visuomotor pathway to the zebrafish optic tectum
Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann
iScience, Cell Press 2026

Figure 6: PrDA neurons exhibit strong synchronized activity

"""

# Imports
import numpy as np
import pandas as pd
from plotting_style import PlotStyle, set_plot_defaults
import matplotlib.pyplot as plt
import matplotlib


def add_fish_identity_bar(ax, fish_ids, y_max=3160):
    if len(fish_ids) == 0:
        return
    n_pairs = len(fish_ids)
    y0_frac = (y_max - n_pairs) / y_max   # fraction of axes where data starts

    unique_fish = list(dict.fromkeys(fish_ids))
    fish_to_value = {fish: i % 2 for i, fish in enumerate(unique_fish)}
    bar = np.array([fish_to_value[f] for f in fish_ids], dtype=float)[:, None]

    ax_bar = ax.inset_axes([1.015, y0_frac, 0.025, 1.0 - y0_frac])  # ← starts at y0_frac
    ax_bar.imshow(bar, aspect="auto", origin="lower", cmap="gray",
                  interpolation="nearest", vmin=0, vmax=1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_frame_on(False)


def plot_coactivity(ax, df, cond, color, method):
    cond_df = df[df['cond'] == cond]

    fish_means = (
        cond_df.groupby(['fish', 'rel_time_s'])[method]
        .mean()
        .reset_index()
        .sort_values('rel_time_s')
    )

    grand = (
        fish_means
        .groupby('rel_time_s')[method]
        .agg(mean='mean', sd='std', n='count')
        .reset_index()
    )
    grand['sem'] = grand['sd'] / np.sqrt(grand['n'])

    for fish, fdf in fish_means.groupby('fish'):
        ax.plot(
            fdf['rel_time_s'], fdf[method],
            color=color, alpha=0.25, lw=0.8
        )

    ax.plot(
        grand['rel_time_s'], grand['mean'],
        color=color, lw=2.0
    )

    ax.fill_between(
        grand['rel_time_s'],
        grand['mean'] - grand['sem'],
        grand['mean'] + grand['sem'],
        color=color,
        alpha=0.2,
        linewidth=0
    )

    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_xlim(cond_df['rel_time_s'].min(), cond_df['rel_time_s'].max())
    ax.set_ylim(0, 1)

    idx = grand['mean'] == grand['mean'].max()
    # print(f'{cond}: max={grand["mean"][idx].item()} at {grand["rel_time_s"][idx].item()}')


def plot_heatmap(ax, lags_s, c_data, cmap="coolwarm", vmin=-1, vmax=1, y_max=3160):
    n_pairs = c_data.shape[0]
    im = ax.imshow(c_data, aspect="auto", origin="lower",
                   extent=[lags_s[0], lags_s[-1], y_max - n_pairs, y_max],  # ← anchored to top
                   vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    ax.axvline(0, color="k", lw=0.8)
    return im


def create_figure():
    # ==================================================================================================================
    # Load Data
    base_dir = 'data/fig6'
    corr_baseline = pd.read_csv(f'{base_dir}/pooled_quiet_spont.csv')
    corr_spont_swim = pd.read_csv(f'{base_dir}/pooled_spont_swim.csv')
    corr_vis_only = pd.read_csv(f'{base_dir}/pooled_visual_noswim.csv')
    corr_vis_swim = pd.read_csv(f'{base_dir}/pooled_visual_swim.csv')
    corr_shuffle_windowed = pd.read_csv(f'{base_dir}/pooled_shuffle_control.csv')
    co_activity = pd.read_csv(f'{base_dir}/zscore_population_activity_timecourses.csv')

    # ==================================================================================================================
    # Figure Settings
    set_plot_defaults()
    style = PlotStyle()

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
    gs_right = left + 0.90 * (right - left)
    gs_bottom = bottom + 0.05 * (top - bottom)
    gs_top = bottom + 0.95 * (top - bottom)

    grid = matplotlib.gridspec.GridSpec(
        nrows=int(content_h_cm * 10), ncols=int(content_w_cm * 10),
        left=gs_left, right=gs_right, bottom=gs_bottom, top=gs_top,
        wspace=0, hspace=0,
    )

    # ------------------------------------------------------------------------------------------------------------------
    axs = dict()
    # Define axes
    # Cross correlation
    height = 0
    size = 40
    size2 = 10
    h_gap = 3
    gap = 5  # mm
    panel_w = int((150 - 4 * gap) / 5)
    x0 = 0
    axs['corr_baseline'] = plt.subplot(grid[height:height + size, x0:x0 + panel_w])
    axs['corr_baseline_mean'] = plt.subplot(grid[height + size + h_gap:height + size + size2 + h_gap, x0:x0 + panel_w])
    x0 += panel_w + gap
    axs['corr_spontan'] = plt.subplot(grid[height:height + size, x0:x0 + panel_w])
    axs['corr_spontan_mean'] = plt.subplot(grid[height + size + h_gap:height + size + size2 + h_gap, x0:x0 + panel_w])
    x0 += panel_w + gap
    axs['corr_vis'] = plt.subplot(grid[height:height + size, x0:x0 + panel_w])
    axs['corr_vis_mean'] = plt.subplot(grid[height + size + h_gap:height + size + size2 + h_gap, x0:x0 + panel_w])
    x0 += panel_w + gap
    axs['corr_vis_swim'] = plt.subplot(grid[height:height + size, x0:x0 + panel_w])
    axs['corr_vis_swim_mean'] = plt.subplot(grid[height + size + h_gap:height + size + size2 + h_gap, x0:x0 + panel_w])
    x0 += panel_w + gap
    axs['corr_shuffle'] = plt.subplot(grid[height:height + size, x0:x0 + panel_w])
    axs['corr_shuffle_mean'] = plt.subplot(grid[height + size + h_gap:height + size + size2 + h_gap, x0:x0 + panel_w])
    axs['corr_cb'] = plt.subplot(grid[5:35, 153:155])

    # Co-activity
    height = 80
    size = 30
    gap = 5  # mm
    panel_w = int((160 - 2 * gap) / 3)
    x0 = 0
    axs['co_activity_sp'] = plt.subplot(grid[height:height+size, x0:x0 + panel_w])
    x0 += panel_w + gap
    axs['co_activity_vis'] = plt.subplot(grid[height:height+size, x0:x0 + panel_w])
    x0 += panel_w + gap
    axs['co_activity_vis_swim'] = plt.subplot(grid[height:height+size, x0:x0 + panel_w])

    # Heatmap settings
    v_min = -1
    v_max = 1
    cmap = 'coolwarm'

    # ==================================================================================================================
    # Axis settings
    y_mc_lim = (-0.5, 1.0)
    y_mc = [0.0, 1.0]
    x_lags = [-5, 0, 5]
    x_lags_lim = (-9, 9)

    ribbon_alpha = 0.1

    # ==================================================================================================================
    # Panel A: Cross Correlations
    # Cross correlation spont. swims
    ax = axs['corr_spontan']
    lags_s = corr_spont_swim.drop(columns=['fish', 'sweep', 'pair_idx']).columns.astype(float).values
    c_data = corr_spont_swim.drop(columns=['fish', 'sweep', 'pair_idx']).values.astype(float)
    plot_heatmap(ax, lags_s, c_data, cmap=cmap, vmin=v_min, vmax=v_max)
    fish_ids = corr_spont_swim['fish'].values.astype(str)
    add_fish_identity_bar(ax, fish_ids)
    ax.set_xlim(-9, 9)
    ax.set_xticks([-5, 0, 5])
    ax.set_ylim(0, 3160)
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels('')
    ax.set_title('Spont. Swims', color=style.motor_color)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([])

    ax = axs['corr_spontan_mean']
    mean = np.mean(c_data, axis=0)
    sd = np.std(c_data, axis=0, ddof=1)
    ax.plot(lags_s, mean, color='black')
    ax.fill_between(lags_s, mean - sd, mean + sd, color='black', alpha=ribbon_alpha)
    ax.set_ylim(y_mc_lim)
    ax.set_yticks(y_mc)
    ax.set_xlim(x_lags_lim)
    ax.set_xticks(x_lags)
    ax.set_yticklabels('')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Lag [s]')

    # ==================================================================================================================
    # Cross correlation Baseline
    ax = axs['corr_baseline']
    lags_s = corr_baseline.drop(columns=['fish', 'sweep', 'pair_idx']).columns.astype(float).values
    c_data = corr_baseline.drop(columns=['fish', 'sweep', 'pair_idx']).values.astype(float)
    plot_heatmap(ax, lags_s, c_data, cmap=cmap, vmin=v_min, vmax=v_max)
    fish_ids = corr_baseline['fish'].values.astype(str)
    add_fish_identity_bar(ax, fish_ids)
    ax.set_xlim(-9, 9)
    ax.set_xticks([-5, 0, 5])
    ax.set_ylim(0, 3160)
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_title('Baseline', color='gray')
    ax.set_ylabel('Neuron pair')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([])

    ax = axs['corr_baseline_mean']
    mean = np.mean(c_data, axis=0)
    sd = np.std(c_data, axis=0, ddof=1)
    ax.plot(lags_s, mean, color='black')
    ax.fill_between(lags_s, mean - sd, mean + sd, color='black', alpha=ribbon_alpha)
    ax.set_ylim(y_mc_lim)
    ax.set_yticks(y_mc)
    ax.set_xlim(x_lags_lim)
    ax.set_xticks(x_lags)
    ax.set_ylabel('Corr.')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Lag [s]')

    # ==================================================================================================================
    # Cross correlation visual (no swim)
    ax = axs['corr_vis']
    lags_s = corr_vis_only.drop(columns=['fish', 'sweep', 'pair_idx']).columns.astype(float).values
    c_data = corr_vis_only.drop(columns=['fish', 'sweep', 'pair_idx']).values.astype(float)
    plot_heatmap(ax, lags_s, c_data, cmap=cmap, vmin=v_min, vmax=v_max)
    fish_ids = corr_vis_only['fish'].values.astype(str)
    add_fish_identity_bar(ax, fish_ids)
    ax.set_xlim(-9, 9)
    ax.set_xticks([-5, 0, 5])
    ax.set_ylim(0, 3160)
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels('')
    ax.set_title('Visual', color=style.stimulus_color)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([])

    ax = axs['corr_vis_mean']
    mean = np.mean(c_data, axis=0)
    sd = np.std(c_data, axis=0, ddof=1)
    ax.plot(lags_s, mean, color='black')
    ax.fill_between(lags_s, mean - sd, mean + sd, color='black', alpha=ribbon_alpha)
    ax.set_ylim(y_mc_lim)
    ax.set_yticks(y_mc)
    ax.set_xlim(x_lags_lim)
    ax.set_xticks(x_lags)
    ax.set_yticklabels('')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Lag [s]')

    # ==================================================================================================================
    # Cross correlation visual + swim
    ax = axs['corr_vis_swim']
    lags_s = corr_vis_swim.drop(columns=['fish', 'sweep', 'pair_idx']).columns.astype(float).values
    c_data = corr_vis_swim.drop(columns=['fish', 'sweep', 'pair_idx']).values.astype(float)
    plot_heatmap(ax, lags_s, c_data, cmap=cmap, vmin=v_min, vmax=v_max)
    fish_ids = corr_vis_swim['fish'].values.astype(str)
    add_fish_identity_bar(ax, fish_ids)
    ax.set_xlim(-9, 9)
    ax.set_xticks([-5, 0, 5])
    ax.set_ylim(0, 3160)
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels('')
    ax.set_title('Visual+Swim', color='tab:orange')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([])

    ax = axs['corr_vis_swim_mean']
    mean = np.mean(c_data, axis=0)
    sd = np.std(c_data, axis=0, ddof=1)
    ax.plot(lags_s, mean, color='black')
    ax.fill_between(lags_s, mean - sd, mean + sd, color='black', alpha=ribbon_alpha)
    ax.set_ylim(y_mc_lim)
    ax.set_yticks(y_mc)
    ax.set_xlim(x_lags_lim)
    ax.set_xticks(x_lags)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Lag [s]')

    # ==================================================================================================================
    # Cross correlation shuffle
    ax = axs['corr_shuffle']
    lags_s = corr_shuffle_windowed.drop(columns=['fish', 'sweep', 'pair_idx']).columns.astype(float).values
    c_data = corr_shuffle_windowed.drop(columns=['fish', 'sweep', 'pair_idx']).values.astype(float)
    plot_heatmap(ax, lags_s, c_data, cmap=cmap, vmin=v_min, vmax=v_max)
    fish_ids = corr_shuffle_windowed['fish'].values.astype(str)
    add_fish_identity_bar(ax, fish_ids)
    ax.set_xlim(-9, 9)
    ax.set_xticks([-5, 0, 5])
    ax.set_ylim(0, 3160)
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels('')
    ax.set_title('Shuffled', color='black')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([])

    ax = axs['corr_shuffle_mean']
    mean = np.mean(c_data, axis=0)
    sd = np.std(c_data, axis=0, ddof=1)
    ax.plot(lags_s, mean, color='black')
    ax.fill_between(lags_s, mean - sd, mean + sd, color='black', alpha=ribbon_alpha)
    ax.set_ylim(y_mc_lim)
    ax.set_yticks(y_mc)
    ax.set_xlim(x_lags_lim)
    ax.set_xticks(x_lags)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Lag [s]')

    # Colorbar
    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    cb = matplotlib.colorbar.ColorbarBase(axs['corr_cb'], norm=norm, cmap=cmap)
    cb.ax.tick_params(labelsize=style.cbar_ticks_size)
    cb.ax.text(7.0, 0.5, f'Corr.', transform=cb.ax.transAxes, rotation=270, ha='center', va='center',
               size=style.cbar_label_size)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

    # ==================================================================================================================
    # Panel B: Spont. Swims Co-activity
    method = 'fraction_active'
    ylim = (0, 1)
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax = axs['co_activity_sp']
    plot_coactivity(ax, co_activity, 'spont_swim', style.motor_color, method=method)
    ax.set_xlim(-5, 15)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_ylim(ylim)
    ax.set_yticks(y_ticks)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Fraction of active neurons')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Spont. Swims', color=style.motor_color)

    # ==================================================================================================================
    # Visual Co-activity
    ax = axs['co_activity_vis']
    plot_coactivity(ax, co_activity, 'visual_noswim', style.stimulus_color, method=method)
    ax.set_xlim(-5, 15)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_ylim(ylim)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels('')
    ax.set_xlabel('Time [s]')
    ax.set_title('Visual', color=style.stimulus_color)
    ax.spines[['top', 'right']].set_visible(False)

    # ==================================================================================================================
    # Visual + Swims Co-activity
    ax = axs['co_activity_vis_swim']
    plot_coactivity(ax, co_activity, 'visual_swim', 'tab:orange', method=method)
    ax.set_xlim(-5, 15)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_ylim(ylim)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels('')
    ax.set_xlabel('Time [s]')
    ax.set_title('Visual+Swim', color='tab:orange')
    ax.spines[['top', 'right']].set_visible(False)

    # ==================================================================================================================
    # Add Sub-Figure Caps
    # from plotting_utils import set_caption_labels
    # set_caption_labels(fig)
    # plt.show()
    # exit()

    cap_pos = {
        'A': (np.float64(0.05), np.float64(0.97)),
        'B': (np.float64(0.05), np.float64(0.40)),
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
    fig_nr = 6

    # Add figure name to pdf
    fig.text(style.figure_label_pos[0], style.figure_label_pos[1], f'Figure {fig_nr}', **style.figure_label)
    plt.savefig(f'{fig_dir}/06_figure_{fig_nr}.pdf', dpi=600)

    plt.close(fig)
    print(f'\n=== STORED FIGURE {fig_nr} TO HDD ====\n')


def main():
    create_figure()


if __name__ == "__main__":
    main()

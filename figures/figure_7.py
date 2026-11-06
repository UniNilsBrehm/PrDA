"""
Assemble and export Figure 7 for the manuscript:
A dopaminergic visuomotor pathway to the zebrafish optic tectum
Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann
iScience, Cell Press 2026

Figure 7: Correcting direction selectivity of PrDA neurons by including motor activity

"""

# Imports
import numpy as np
import pandas as pd
from plotting_style import PlotStyle, set_plot_defaults
from plotting_utils import hide_axis_spines, add_pictogram
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import kstest


def get_direction_selectivity_data(base_dir):
    direction_selectivity = pd.read_csv(f'{base_dir}/direction_selectivity.csv', index_col=0)
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

    # Store to HDD
    results.to_csv(f'{base_dir}/ds_results.csv')
    ks_results.to_csv(f'{base_dir}/ds_ks_test.csv')
    # print('DS data stored to HDD')


def load_direction_selectivity_data(base_dir):
    ds_data = pd.read_csv(f'{base_dir}/ds_results.csv', index_col=0)
    ks_results = pd.read_csv(f'{base_dir}/ds_ks_test.csv', index_col=0)
    return ds_data, ks_results


def create_figure():
    set_plot_defaults()
    style = PlotStyle()

    # Load Direction Selectivity Data
    base_dir = 'data/fig7'
    ds_data, ks_results = load_direction_selectivity_data(base_dir)
    ds = ds_data['ds_stimulus']
    ds_swim = ds_data['ds_swim']
    ds_corrected = ds_data['ds_corrected']

    # ==================================================================================================================
    # Figure
    # A4 page: The figure will be on a DinA4 page in the final pdf file
    a4_w_cm, a4_h_cm = 21.0, 29.7
    fig_width = a4_w_cm / 2.54
    fig_height = a4_h_cm / 2.54

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Center the 16x20 cm content block on the A4 page
    content_w_cm, content_h_cm = 6, 15.5
    left = ((a4_w_cm - content_w_cm) / 2) / a4_w_cm
    right = 1 - left
    bottom = ((a4_h_cm - content_h_cm) / 2) / a4_h_cm
    top = 1 - bottom

    gs_left = left + 0.05 * (right - left)
    gs_right = left + 0.95 * (right - left)
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
    # Direction Selectivity
    axs['ds_swim'] = plt.subplot(grid[0:30, 10:60])
    axs['ds_stim'] = plt.subplot(grid[60:90, 10:60])
    axs['ds_corrected'] = plt.subplot(grid[120:150, 10:60])

    # ==================================================================================================================
    # Direction selectivity index distributions
    # Add pictograms
    pic_size = 0.012
    sym_fw = plt.imread(f'{base_dir}/images/symbol_grating_0.png')
    sym_bw = plt.imread(f'{base_dir}/images/symbol_grating_180.png')
    txt_y = 1.10
    pic_y = 0.95

    # ------------------------------------------------------------------------------------------------------------------
    # A) direction selectivity index distribution: Swims (behavioral response)
    # Remove consecutive duplicates
    ds_swim = ds_swim[ds_swim != ds_swim.shift()]
    bins_ds = 7
    median_val = np.median(ds_swim)
    panel = 'ds_swim'
    axs[panel].hist(ds_swim, bins=bins_ds, density=False, edgecolor='black', facecolor=style.motor_color, alpha=1)
    axs[panel].axvline(0, **style.lsDS_Zero)
    axs[panel].set_xlim(-1.1, 1.1)
    axs[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axs[panel].set_xlabel(r'DSI$_{swim}$')
    axs[panel].set_ylim(0, 10)
    axs[panel].set_yticks([0, 5, 10])
    axs[panel].set_ylabel("Count (sweeps)")
    axs[panel].text(0.25, txt_y, 'Backward', transform=axs[panel].transAxes, ha='center', va='center', **style.txtAnnotation)
    axs[panel].text(0.75, txt_y, 'Forward', transform=axs[panel].transAxes, ha='center', va='center', **style.txtAnnotation)
    add_pictogram(sym_bw, (0.25, pic_y), axs[panel], size=pic_size, ha='center', va='top', coord='axes fraction')
    add_pictogram(sym_fw, (0.75, pic_y), axs[panel], size=pic_size, ha='center', va='top', coord='axes fraction')
    axs[panel].text(0.05, 0.6, f'M={median_val:.2f}', transform=axs[panel].transAxes, ha='left', va='center', **style.txtAnnotation)
    hide_axis_spines(axs[panel])

    # ------------------------------------------------------------------------------------------------------------------
    # B) direction selectivity index distribution: Stimulus Regressor
    bins_ds = 15
    y_lim = 100
    median_val = np.median(ds)
    panel = 'ds_stim'
    axs[panel].hist(ds, bins=bins_ds, density=False, edgecolor='black', facecolor=style.stimulus_color, alpha=1)
    axs[panel].axvline(0, **style.lsDS_Zero)
    axs[panel].set_xlim(-1.1, 1.1)
    axs[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axs[panel].set_xlabel(r'DSI$_{neuron, apparent}$')
    axs[panel].set_ylim(0, y_lim)
    axs[panel].set_ylabel("Count (neurons)")
    axs[panel].set_yticks([0, 25, 50, 75, 100])
    axs[panel].text(0.25, txt_y, 'Backward', transform=axs[panel].transAxes, ha='center', va='center', **style.txtAnnotation)
    axs[panel].text(0.75, txt_y, 'Forward', transform=axs[panel].transAxes, ha='center', va='center', **style.txtAnnotation)
    add_pictogram(sym_bw, (0.25, pic_y), axs[panel], size=pic_size, ha='center', va='top', coord='axes fraction')
    add_pictogram(sym_fw, (0.75, pic_y), axs[panel], size=pic_size, ha='center', va='top', coord='axes fraction')
    axs[panel].text(0.05, 0.6, f'M={median_val:.2f}', transform=axs[panel].transAxes, ha='left', va='center', **style.txtAnnotation)

    hide_axis_spines(axs[panel])
    # C) direction selectivity index distribution: Corrected
    bins_ds = 15
    y_lim = 100
    median_val = np.median(ds_corrected)
    p_val = ks_results[ks_results['ds'] == 'corrected']['p'].item()
    panel = 'ds_corrected'
    axs[panel].hist(ds_corrected, bins=bins_ds, density=False, edgecolor='black', facecolor='tab:gray', alpha=1)
    axs[panel].axvline(0, **style.lsDS_Zero)
    axs[panel].set_xlim(-1.1, 1.1)
    axs[panel].set_xticks([-1, -0.5, 0, 0.5, 1])
    axs[panel].set_xlabel(r'DSI$_{neuron, corrected}$')
    axs[panel].set_ylim(0, y_lim)
    axs[panel].set_ylabel("Count (neurons)")
    axs[panel].set_yticks([0, 25, 50, 75, 100])
    axs[panel].text(0.25, txt_y, 'Backward', transform=axs[panel].transAxes, ha='center', va='center', **style.txtAnnotation)
    axs[panel].text(0.75, txt_y, 'Forward', transform=axs[panel].transAxes, ha='center', va='center', **style.txtAnnotation)
    add_pictogram(sym_bw, (0.25, pic_y), axs[panel], size=pic_size, ha='center', va='top', coord='axes fraction')
    add_pictogram(sym_fw, (0.75, pic_y), axs[panel], size=pic_size, ha='center', va='top', coord='axes fraction')
    axs[panel].text(0.05, 0.6, f'M={median_val:.2f}', transform=axs[panel].transAxes, ha='left', va='center', **style.txtAnnotation)
    hide_axis_spines(axs[panel])

    # ==================================================================================================================
    # Add Sub-Figure Caps
    # from plotting_utils import set_caption_labels
    # set_caption_labels(fig)
    # plt.show()
    # exit()

    cap_pos = {
        'A': (np.float64(0.03), np.float64(0.97)),
        'B': (np.float64(0.03), np.float64(0.64)),
        'C': (np.float64(0.03), np.float64(0.30)),
    }

    # Redefine position to DinA4 page
    def content_xy(fx, fy):
        return left + fx * (right - left), bottom + fy * (top - bottom)

    for cap, (fx, fy) in cap_pos.items():
        x, y = content_xy(fx, fy)
        fig.text(x, y, cap, **style.cap_style)

    # ==================================================================================================================
    # Store figures
    fig_dir = f"figs"
    matplotlib.rcParams["savefig.bbox"] = None
    fig_nr = 7

    # Add figure name to pdf
    fig.text(style.figure_label_pos[0], style.figure_label_pos[1], f'Figure {fig_nr}', **style.figure_label)

    plt.savefig(f'{fig_dir}/07_figure_{fig_nr}.pdf', dpi=600)
    plt.close(fig)
    print(f'\n=== STORED FIGURE {fig_nr} TO HDD ====\n')


def main():
    # Prepare Data
    base_dir = 'data/fig7'
    get_direction_selectivity_data(base_dir)

    # Create figure
    create_figure()


if __name__ == "__main__":
    main()

"""
Assemble and export Figure 3 for the manuscript:
A dopaminergic visuomotor pathway to the zebrafish optic tectum
Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann
iScience, Cell Press 2026

Figure 3: Spontaneous motor events are associated with phasic PrDA neuron activity

"""

# Imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from plotting_style import PlotStyle, set_plot_defaults
from plotting_utils import hide_axis_spines, add_scale_bar_v2
from utils import calcium_impulse_response, create_regressors_from_binary, merge_events_closing


def prepare_data():
    # ------------------------------------------------------------------------------------------------------------------
    # save_dir = f'{Config.BASE_DIR}/figures/data/spont_motor/'

    # Load Data
    scores_org = pd.read_csv(f'data/fig3/linear_scoring_results.csv')
    # ------------------------------------------------------------------------------------------------------------------
    # Prepare Data
    # Remove not needed entries from score data set
    idx = scores_org['reg'].isin(
        ['motor_spontaneous_MEAN', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor',
         'moving_target_02_motor'])
    df_scores = scores_org[~idx].reset_index(drop=True)

    # Get spontaneous swim events data
    df_sp = df_scores[df_scores['reg'] == 'motor_spontaneous'].copy().reset_index(drop=True)
    df_sp["event_idx"] = df_sp.groupby("roi").cumcount()

    df_sub_sp = df_sp[
        ['roi', 'sw', 'fish', 'score_norm99', 'score', 'r2', 'coeff', 'lag_sec', 'threshold', 'threshold_r2',
         'threshold_slope', 'significant', 'event_idx']].copy()

    df_per_roi = (
        df_sub_sp.groupby(["roi"])["significant"]
        .agg(
            n_significant="sum",  # number of True
            total_events="count"  # total rows
        )
        .assign(fraction=lambda x: x["n_significant"] / x["total_events"])
        .reset_index()
    )


    # Prepare Data for Cumulative Histograms:
    # - Number of significant events
    # - Fraction of significant events (response probability)
    # Remove all neurons that did see less than 3 spont. motor events
    event_th = 3
    idx = df_per_roi['total_events'] < event_th
    df_per_roi_selected = df_per_roi[~idx].copy()
    data_cum = df_per_roi_selected[["n_significant", "fraction"]].copy()

    # EXPORT to HDD
    data_cum.to_csv(f'data/fig3/spont_motor/data_cumulative_response_prob.csv')

    # ------------------------------------------------------------------------------------------------------------------
    # Anatomical positions and scores
    data_set = df_sp.copy()
    data_set.loc[data_set['score'] < 0, 'score'] = 0
    data_set.loc[data_set['score_norm99'] < 0, 'score_norm99'] = 0

    a_data = pd.DataFrame()
    a_data['score'] = data_set.groupby(['roi'])['score'].agg(mean="median")
    a_data['score_norm99'] = data_set.groupby(['roi'])['score_norm99'].agg(mean="median")
    a_data['fish'] = data_set.groupby(['roi'])['fish'].agg(mean="median")
    a_data['x'] = data_set.groupby(['roi'])['x_position'].agg(x='unique')
    a_data['x'] = a_data['x'].str[0]
    a_data['y'] = data_set.groupby(['roi'])['y_position'].agg(y='unique')
    a_data['y'] = a_data['y'].str[0]
    a_data['z'] = data_set.groupby(['roi'])['z_position'].agg(z='unique')
    a_data['z'] = a_data['z'].str[0]

    a_data.reset_index(inplace=True)
    a_data['n_significant'] = df_per_roi['n_significant']
    a_data['n_total'] = df_per_roi['total_events']
    a_data['fraction'] = df_per_roi['fraction']

    # EXPORT to HDD
    a_data.to_csv(f'data/fig3/spont_motor/anatomical_data.csv')


def plot_main_figure():
    set_plot_defaults()
    style = PlotStyle()

    fig_data = "fig3"

    # Load Data
    c_data = pd.read_csv(f'data/{fig_data}/spont_motor/data_cumulative_response_prob.csv', index_col=0)
    a_data = pd.read_csv(f'data/{fig_data}/spont_motor/anatomical_data.csv', index_col=0)

    # Load Data
    ca_sampling_rate_file = f'data/{fig_data}/sampling_rate.csv'
    ca_df_f_file = f'data/{fig_data}/df_f_data.csv'
    ca_df_f = pd.read_csv(ca_df_f_file)
    ca_sampling_rate = pd.read_csv(ca_sampling_rate_file)['sampling_rate'].item()
    ca_time_axis = np.linspace(0, ca_df_f.shape[0] / ca_sampling_rate, ca_df_f.shape[0])

    m_sc, m_r2, m_slope, m_r = get_sp_motor_scores()

    # VR Recordings
    vr_sampling_rate = 10000

    # Prepare data
    # Example 1
    # sw = 'sw_20'
    # rois = ['285']
    sw = 'sw_31'
    rois = ['457']

    ca_trace = ca_df_f[rois[0]].to_numpy()
    vr_rec = pd.read_hdf(f'data/{fig_data}/vr_rec_{sw}.h5', key='data')

    vr_trace = vr_rec.to_numpy()
    vr_time_axis = np.linspace(0, vr_trace.shape[0] / vr_sampling_rate, vr_trace.shape[0])
    vr_events = pd.read_csv(f'data/{fig_data}/example_1_vr_events.csv', index_col=0)

    vr_binary = np.zeros(ca_df_f.shape[0])
    vr_binary[vr_events['start_idx'].to_numpy()] = 1
    vr_binary_merged = merge_events_closing(vr_binary, int(8 * ca_sampling_rate))

    _, cif = calcium_impulse_response(
        tau_rise=3, tau_decay=6, amplitude=1.0, sampling_rate=ca_sampling_rate, threshold=1e-5, norm=True
    )
    vr_reg = create_regressors_from_binary(vr_binary_merged, cif, delta=True, norm=False)

    # ==================================================================================================================
    # Figure
    # A4 page: The figure will be on a DinA4 page in the final pdf file
    a4_w_cm, a4_h_cm = 21.0, 29.7
    fig_width = a4_w_cm / 2.54
    fig_height = a4_h_cm / 2.54

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Center the 16x20 cm content block on the A4 page
    content_w_cm, content_h_cm = 14, 12
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
    axs['vr_trace'] = plt.subplot(grid[0:12, 0:22])
    axs['vr_reg'] = plt.subplot(grid[14:26, 0:22])
    axs['ca_trace'] = plt.subplot(grid[28:40, 0:22])

    # Corr and Score histogram
    axs['corr_hist'] = plt.subplot(grid[0:35, 45:85])
    axs['score_hist'] = plt.subplot(grid[0:35, 100:140])

    # Response Prob.
    axs['fraction_sign'] = plt.subplot(grid[55:95, 15:55])

    # Anatomical Map
    axs['anatomical_map'] = plt.subplot(grid[55:95, 71:129])  # 40 x 60
    axs['anatomical_map_colorbar'] = plt.subplot(grid[65:85, 129:130])

    # ------------------------------------------------------------------------------------------------------------------
    # A) Example 1
    axs['vr_trace'].plot(vr_time_axis, vr_trace, 'gray')
    axs['vr_reg'].plot(ca_time_axis, vr_reg, color=style.motor_color)
    axs['ca_trace'].plot(ca_time_axis, ca_trace, 'k')

    # Add Annotations
    axs['vr_trace'].text(0, -0.05, 'MNR', transform=axs['vr_trace'].transAxes, ha='left', va='center',
                         **style.txtVRLabel)
    axs['vr_reg'].text(0, -0.05, 'Reg', transform=axs['vr_reg'].transAxes, ha='left', va='center',
                       **style.txtMotorLabel)
    axs['ca_trace'].text(0, -0.2, f'{style.deltaF}', transform=axs['ca_trace'].transAxes, ha='left', va='center',
                         **style.txtCaTraceLabel)

    # Label sign events
    x_pos_sign = [69, 165, 196, 270, 335, 407]

    for x_pos in x_pos_sign:
        axs['ca_trace'].axvline(x=x_pos-5, color="gray", linestyle="--")

    # ..................................................................................................................
    # X Limit
    x_lim_min = 12  # secs
    x_lim_max = 300
    for ax_name in ['vr_trace', 'vr_reg', 'ca_trace']:
        axs[ax_name].set_xlim(x_lim_min, x_lim_max)
        # Remove Axes
        axs[ax_name].axis('off')

    # Y Limit
    axs['vr_trace'].set_ylim(-10, 10)
    axs['vr_reg'].set_ylim(-0.2, 1.5)
    axs['ca_trace'].set_ylim(-0.2, 5)

    # Add Scale Bars
    # Time Scale Bar
    x_pos = 1.1
    y_pos = 0.1
    lw = 2
    add_scale_bar_v2(
        axs['ca_trace'], size=30, label='', location=(x_pos, y_pos), orientation='horizontal', color='black',
        linewidth=lw, fontsize=10, padding=0.1)

    # Delta F over F Scale Bar
    add_scale_bar_v2(
        axs['ca_trace'], size=1, label=f'', location=(x_pos, y_pos), orientation='vertical',
        color='black',
        linewidth=lw, fontsize=10, padding=0.02)

    # ------------------------------------------------------------------------------------------------------------------
    # Corr and Score histogram
    axs['corr_hist'].hist(m_r, bins=30, **style.histogram_style)
    axs['corr_hist'].set_xlim(-1, 1)
    axs['corr_hist'].set_xticks([-1, -0.5, 0, 0.5, 1])
    axs['corr_hist'].set_yticks([0, 300, 600])
    axs['corr_hist'].set_ylim(0, 600)
    axs['corr_hist'].set_xlabel('Correlation')
    axs['corr_hist'].set_ylabel('Count')

    axs['score_hist'].hist(m_sc, bins=40, **style.histogram_style)
    axs['score_hist'].set_xlim(-1, 4)
    axs['score_hist'].set_xticks([-1, 0, 1, 2, 3, 4])
    axs['score_hist'].set_yticks([0, 600, 1200])
    axs['score_hist'].set_ylim(0, 1200)
    axs['score_hist'].set_xlabel('Response score')

    # ------------------------------------------------------------------------------------------------------------------
    # Fraction of sign. events (response prob)
    # Compute ECDFs
    data = c_data['fraction']
    color = style.fig_3_hist_prob_color
    ecdf_data = ECDF(data)
    x_vals_data = np.sort(data)

    axs['fraction_sign'].plot(x_vals_data, ecdf_data(x_vals_data), '.-', color=color, label='Fraction', alpha=1)
    axs['fraction_sign'].fill_between(x_vals_data, ecdf_data(x_vals_data), color=color, alpha=0.1)
    axs['fraction_sign'].set_xlabel("Response prob. [%]")
    axs['fraction_sign'].set_xlim(0, 1.1)
    axs['fraction_sign'].set_xticks([0.0, 0.25, 0.5, 0.75, 1.00])
    axs['fraction_sign'].set_xticklabels(["0", "25", "50", "75", "100"])

    axs['fraction_sign'].set_ylabel("Cumulative frequency")
    axs['fraction_sign'].set_ylim(0, 1.1)
    axs['fraction_sign'].set_yticks([0.0, 0.25, 0.5, 0.75, 1.00])
    axs['fraction_sign'].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Add grid
    axs['fraction_sign'].grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)

    # Highlight certain points (in text)
    idx = np.where(ecdf_data(x_vals_data) >= 0.25)[0][0]
    res_prob = x_vals_data[idx - 1]
    axs['fraction_sign'].plot([0, res_prob], [0.25, 0.25], **style.fig_3_mark_style)
    axs['fraction_sign'].plot([res_prob, res_prob], [0, 0.25], **style.fig_3_mark_style)

    # Highlight certain points (in text)
    axs['fraction_sign'].plot([0, 0.5], [0.5, 0.5], **style.fig_3_mark_style_dashed)
    axs['fraction_sign'].plot([0.5, 0.5], [0, 0.5], **style.fig_3_mark_style_dashed)

    # ------------------------------------------------------------------------------------------------------------------
    # C) Anatomical Map
    scale_factor = 0.131  # turning px in micro meter
    v_min = 0
    v_max = 1
    x_midline = (a_data['y'].max() + a_data['y'].min()) / 2
    y_midline = (a_data['x'].max() + a_data['x'].min()) / 2

    axs['anatomical_map'].scatter(
        (a_data['y'] - x_midline) * scale_factor,  # now x-axis
        (a_data['x'] - y_midline) * scale_factor,  # now y-axis
        c=a_data['fraction'],
        vmin=v_min,
        vmax=v_max,
        **style.scAnatomical_scatter
    )
    axs['anatomical_map'].axvline(x=0, color='k', linestyle='--', linewidth=1)  # midline becomes vertical
    axs['anatomical_map'].set_xlabel(r'Distance to midline [$\mu$m]')
    axs['anatomical_map'].set_ylabel(r'Anteroposterior axis [$\mu$m]')
    axs['anatomical_map'].set_aspect('equal', adjustable='box')
    axs['anatomical_map'].set_xticks([-10, -5, 0, 5, 10])
    axs['anatomical_map'].set_yticks([-10, -5, 0, 5, 10])

    axs['anatomical_map'].text(1.0, 0.99, 'Anterior', transform=axs['anatomical_map'].transAxes, ha='right',
                               va='center', **style.txtCaTraceLabel)
    axs['anatomical_map'].text(1.0, 0.05, 'Posterior', transform=axs['anatomical_map'].transAxes, ha='right',
                               va='center', **style.txtCaTraceLabel)

    # Colorbar
    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    cb = matplotlib.colorbar.ColorbarBase(axs['anatomical_map_colorbar'], norm=norm, cmap=style.anatomical_cmap,
                                          orientation='vertical')
    cb.ax.tick_params(labelsize=style.cbar_ticks_size)
    cb.set_label('Response prob.', rotation=270, labelpad=10, size=8)
    cb.set_ticks([0.0, 0.5, 1.0])

    # ------------------------------------------------------------------------------------------------------------------
    # Hide top and right axis spines
    hide_axis_spines(axs['fraction_sign'])
    hide_axis_spines(axs['anatomical_map'])
    hide_axis_spines(axs['corr_hist'])
    hide_axis_spines(axs['score_hist'])

    # ==================================================================================================================
    # Add Sub-Figure Caps
    # from plotting_utils import set_caption_labels
    # set_caption_labels(fig)
    # plt.show()
    # exit()

    cap_pos = {
        'A': (np.float64(0.18), np.float64(0.70)),
        'B': (np.float64(0.33), np.float64(0.70)),
        'C': (np.float64(0.58), np.float64(0.70)),
        'D': (np.float64(0.18), np.float64(0.52)),
        'E': (np.float64(0.48), np.float64(0.52)),
    }

    for cap, (fx, fy) in cap_pos.items():
        x, y = fx, fy
        fig.text(x, y, cap, **style.cap_style)

    # ==================================================================================================================
    # Store Figure to HDD
    fig_dir = f"figs"
    matplotlib.rcParams["savefig.bbox"] = None
    fig_nr = 3

    # Add figure name to pdf
    fig.text(style.figure_label_pos[0], style.figure_label_pos[1], f'Figure {fig_nr}', **style.figure_label)
    plt.savefig(f'{fig_dir}/03_figure_{fig_nr}.pdf', dpi=600)

    plt.close(fig)
    print(f'\n=== STORED FIGURE {fig_nr} TO HDD ====\n')


def get_sp_motor_scores():
    # Load Data
    scores_org = pd.read_csv(f'data/fig3/linear_scoring_results.csv')

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare Data
    # Remove not needed entries from score data set
    idx = scores_org['reg'].isin(
        ['motor_spontaneous_MEAN', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor',
         'moving_target_02_motor'])
    df_scores = scores_org[~idx].reset_index(drop=True)

    # Get spontaneous swim events data
    df_sp = df_scores[df_scores['reg'] == 'motor_spontaneous'].copy().reset_index(drop=True)
    df_sp["event_idx"] = df_sp.groupby("roi").cumcount()

    sc = df_sp['score'].to_numpy()
    r2 = df_sp['r2'].to_numpy()
    slope = df_sp['coeff'].to_numpy()

    # extract pearsons corr.
    r = np.sign(slope) * np.sqrt(r2)

    return sc, r2, slope, r


def main():
    # Prepare Data
    prepare_data()

    # Create Figure
    plot_main_figure()


if __name__ == "__main__":
    main()

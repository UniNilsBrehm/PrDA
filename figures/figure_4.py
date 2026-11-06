"""
Assemble and export Figure 4 for the manuscript:
A dopaminergic visuomotor pathway to the zebrafish optic tectum
Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann
iScience, Cell Press 2026

Figure 4: PrDA neurons are driven by visual stimulation and respond bilaterally

"""
import numpy as np
import pandas as pd
from plotting_style import PlotStyle, set_plot_defaults
from plotting_utils import create_alternating_pattern, hide_axis_spines, remove_x_axis, remove_y_axis, add_pictogram, \
    add_scale_bar, add_scale_bar_v2
import matplotlib.pyplot as plt
import matplotlib

from utils import get_rois_per_sweep, load_hdf5_as_dict, calcium_impulse_response, \
    create_regressors_from_binary
from scipy.stats import mannwhitneyu


def plot_visual_left_right_hist(ax, neuron_df, p_val, style, n_bins=10):
    """
    Histogram of fraction of purely visual stimulus types (significant / seen)
    for left vs right neurons.
    """
    bins = np.linspace(0, 1, n_bins + 1)

    left_vals = neuron_df.loc[neuron_df["side"] == "left", "frac_visual_stimuli"].dropna()
    right_vals = neuron_df.loc[neuron_df["side"] == "right", "frac_visual_stimuli"].dropna()

    color_left = style.left_color
    color_right = style.right_color

    n, bins, patches = ax.hist(
        [left_vals, right_vals],
        bins=bins,
        label=["Left (ipsilateral)", "Right (contralateral)"],
        color=[color_left, color_right],
        alpha=0.8
    )

    # Apply hatching
    hatches = ["///", ""]  # left, right

    for dataset_patches, hatch in zip(patches, hatches):
        for bar in dataset_patches:
            bar.set_edgecolor("black")
            bar.set_linewidth(1)
            bar.set_hatch(hatch)

    ax.set_xlim((0, 1))
    ax.set_xlabel('Fraction of vis. stimuli with response')
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel('Count (neurons)')
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.60, 0.70),  # legend's bottom-left corner goes exactly here
        frameon=False,
        fontsize=8
    )
    ax.text(0.02, 0.95, f'p={p_val:.2f}', transform=ax.transAxes, **style.txtAnnotation)


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

    # No motor + sign. vis response
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

    # Denominator: stimulus types with no significant motor response for this neuron
    seen_counts = (
        merged[~merged["significant_motor"].astype(bool)]
        .groupby("roi", as_index=False)
        .agg({"reg_base": "nunique"})
        .rename(columns={"reg_base": "n_visual_stimuli_seen"})
    )

    neuron_df = pd.merge(all_neurons, visual_counts, on="roi", how="left")
    neuron_df = pd.merge(neuron_df, seen_counts, on="roi", how="left")

    neuron_df["n_visual_stimuli"] = neuron_df["n_visual_stimuli"].fillna(0).astype(int)
    neuron_df["n_visual_stimuli_seen"] = neuron_df["n_visual_stimuli_seen"].fillna(0).astype(int)

    neuron_df["frac_visual_stimuli"] = neuron_df["n_visual_stimuli"].div(
        neuron_df["n_visual_stimuli_seen"].replace(0, np.nan)
    )
    return neuron_df


def get_score_data(base_dir):

    linear_scoring_data = pd.read_csv(
        f'{base_dir}/linear_scoring_results.csv'
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

    # Mann–Whitney U test: Left vs. Right
    left = neuron_df[neuron_df['side'] == 'left']['frac_visual_stimuli']
    right = neuron_df[neuron_df['side'] == 'right']['frac_visual_stimuli']
    stat_frac, p_frac = mannwhitneyu(left, right, alternative='two-sided')

    n1 = len(left)
    n2 = len(right)

    # Rank-biserial correlation
    r_rb = 1 - (2 * stat_frac) / (n1 * n2)

    # print('Mann–Whitney U test: Left vs. Right: Fraction of responses')
    # print('U, p-val, rank-biserial correlation')
    # print(stat_frac, p_frac, r_rb)

    return neuron_df, p, p_frac


def plot_visual_anatomy_binary(ax_map, neuron_df):
    """
    Anatomical map:
    closed = n_visual_stimuli == 0
    open = n_visual_stimuli > 0
    """
    zero = neuron_df["n_visual_stimuli"].eq(0)
    responsive = neuron_df["n_visual_stimuli"].gt(0)

    # v = 0: filled black dots
    ax_map.scatter(
        neuron_df.loc[zero, "x_plot"],
        neuron_df.loc[zero, "y_plot"],
        s=12,
        marker="o",
        facecolors="lightgray",
        edgecolors="tab:gray",
        linewidths=0.5,
        alpha=1,
        label='non-visual'
    )

    # v > 0: green circles
    ax_map.scatter(
        neuron_df.loc[responsive, "x_plot"],
        neuron_df.loc[responsive, "y_plot"],
        s=8,
        marker="o",
        facecolors="tab:green",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
        label='visual'
    )

    ax_map.axvline(
        x=0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_map.set_xlabel(r"Distance to midline [$\mu$m]")
    ax_map.set_ylabel(r"Anteroposterior axis [$\mu$m]")
    ax_map.set_aspect("equal", adjustable="box")

    ax_map.set_xlim(-15, 15)
    ax_map.set_ylim(-15, 15)

    ticks = [-15, -10, -5, 0, 5, 10, 15]
    ax_map.set_xticks(ticks)
    ax_map.set_yticks(ticks)

    ax_map.text(
        1.0,
        0.99,
        "Anterior",
        transform=ax_map.transAxes,
        ha="right",
        va="center",
        fontsize=8,
    )

    ax_map.text(
        1.0,
        0.05,
        "Posterior",
        transform=ax_map.transAxes,
        ha="right",
        va="center",
        fontsize=8,
    )

    legend = ax_map.legend(
        loc="lower left",
        bbox_to_anchor=(0, 0.9),
        frameon=True,
        fontsize=6,
        ncols=1,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        columnspacing=0.5,
        handletextpad=0.05,
        borderpad=0.2,
    )
    legend.get_frame().set_linewidth(0.5)  # e.g., 0.5 pt


def load_heatmap_data(base_dir):
    # Directories
    vr_events_dir = f'{base_dir}/selected_sweeps_ventral_root_events.h5'

    ca_sampling_rate_file = f'{base_dir}/sampling_rate.csv'
    ca_df_f_file = f'{base_dir}/df_f_data.csv'
    ca_df_f_file_no_motor = f'{base_dir}/df_f_data_no_motor.csv'

    ca_labels_file = f'{base_dir}/df_f_data_labels.csv'
    stimulus_trace_dir = f'{base_dir}/stimulus_traces.csv'
    stimulus_onsets_dir = f'{base_dir}/sw_01_stimulus_onsets.csv'

    # Load data for selected sweeps
    ca_df_f = pd.read_csv(ca_df_f_file)
    ca_df_f_no_motor = pd.read_csv(ca_df_f_file_no_motor)
    ca_sampling_rate = pd.read_csv(ca_sampling_rate_file)['sampling_rate'].item()
    ca_time_axis = np.linspace(0, ca_df_f.shape[0] / ca_sampling_rate, ca_df_f.shape[0])

    # Load labels for selected sweeps
    ca_labels = pd.read_csv(ca_labels_file)

    # Load Stimulus
    stimulus_traces = pd.read_csv(stimulus_trace_dir)
    stimulus_trace = stimulus_traces['sw_01']
    stimulus_onsets = pd.read_csv(stimulus_onsets_dir, index_col=0)

    # Detected VR Events for selected data set
    vr_events = load_hdf5_as_dict(vr_events_dir)
    sw_rois = get_rois_per_sweep(ca_labels)

    # Get motor regs from each sweep for heatmap
    sw_list = list(ca_labels.iloc[1, :])
    sw_motor_regs = compute_vr_regressors(vr_events, ca_df_f .iloc[:, 0], ca_sampling_rate)
    sw_motor_regs_heatmap = np.stack([np.asarray(sw_motor_regs[s]) for s in sw_list], axis=0).T  # shape: (360, T)

    data = {
        'ca': ca_df_f,
        'ca_no_motor': ca_df_f_no_motor,
        'ca_time_axis': ca_time_axis,
        'stimulus_trace': stimulus_trace,
        'stimulus_onsets': stimulus_onsets,
        'vr_events': vr_events,
        'vr_reg_map': sw_motor_regs_heatmap,
        'sw_rois': sw_rois,
        'ca_labels': ca_labels,
        'ca_sampling_rate': ca_sampling_rate
    }

    return data


def plot_heatmap(ax, data, number_of_rois, fish_ids_bar, z_plane_bar, data_id, show_me=False):
    t_axis = data['ca_time_axis']
    stimulus_onsets = data['stimulus_onsets']
    vr_events = data['vr_events']
    sw_rois = data['sw_rois']
    try:
        ca_data = data[data_id]
    except KeyError:
        print('ERROR: COULD NOT FIND KEY IN DATA')
        return

    style = PlotStyle()
    v_min = 0
    v_max = 3
    ax.imshow(
        ca_data.T, aspect='auto', extent=(t_axis[0], t_axis[-1], 0, number_of_rois),
        cmap=style.heatmap_cmap, origin='lower', vmin=v_min, vmax=v_max, rasterized=True
    )
    # Add Fish ID Bar
    # [x0, y0, width, height]
    inset_ax = ax.inset_axes([1.006, 0, 0.01, 1], transform=ax.transAxes)
    inset_ax.imshow(fish_ids_bar, aspect='auto', cmap=style.fish_ids_cmap, origin='lower', rasterized=True)
    inset_ax.axis('off')

    # Add Z-Plane Bar
    # [x0, y0, width, height]
    inset_ax = ax.inset_axes([1.015, 0, 0.01, 1], transform=ax.transAxes)
    inset_ax.imshow(z_plane_bar, aspect='auto', cmap=style.z_plane_cmap, origin='lower', rasterized=True)
    inset_ax.axis('off')

    # Plot VR Events on heat map
    # Loop over sweeps
    if show_me:
        start_roi = 0
        for sw in vr_events:
            # Get the Motor Events for this sweep
            motor_onsets = vr_events[sw]['start_time'].round()
            rois_range = len(sw_rois[sw])
            end_roi = start_roi + rois_range
            # Loop over motor events
            for m_onset in motor_onsets:
                ax.plot([m_onset, m_onset], [start_roi, end_roi], **style.motor_onsets)
            start_roi = end_roi

    # Plot Stimulus Onsets as a line in the heat map
    for onset in stimulus_onsets.iloc[0]:
        on_time = np.floor(onset)
        ax.plot([on_time, on_time], [0, number_of_rois + 0.5], **style.lsStimulusOnsetLine_solid)
    ax.set_xlim([0, t_axis[-1]])
    ax.set_ylim([0.5, number_of_rois+0.5])


def get_example_traces(data):
    vis_idx = ["61", "180",  "226"]  # the roi name
    vis_num = ["61", "102", "148"]  # the roi number (0-359)
    vis_sw = ["sw_03", "sw_14", "sw_16"]

    trace_vis = data[vis_idx]

    return trace_vis, vis_sw, vis_idx, vis_num


def add_stimulus_pics(ax, pic_size, base_dir):
    # Load Images
    sym_mt = plt.imread(f'{base_dir}/symbol_mt.jpg')
    sym_dark_loom = plt.imread(f'{base_dir}/symbol_dark_loom.jpg')
    sym_bright_loom = plt.imread(f'{base_dir}/symbol_bright_loom.jpg')
    sym_grating_appears = plt.imread(f'{base_dir}/symbol_grating_appears.jpg')
    sym_grating_0 = plt.imread(f'{base_dir}/symbol_grating_0.jpg')
    sym_grating_180 = plt.imread(f'{base_dir}/symbol_grating_180.jpg')
    sym_dark_flash = plt.imread(f'{base_dir}/symbol_dark_flash.jpg')
    sym_bright_flash = plt.imread(f'{base_dir}/symbol_bright_flash.jpg')

    add_pictogram(sym_mt, (302, 1.05), ax, size=pic_size)
    add_pictogram(sym_mt, (370, 1.05), ax, size=pic_size)
    add_pictogram(sym_grating_appears, (434, 1.05), ax, size=pic_size)
    add_pictogram(sym_grating_0, (500, 1.05), ax, size=pic_size)
    add_pictogram(sym_grating_180, (570, 1.05), ax, size=pic_size)
    add_pictogram(sym_grating_appears, (634, 1.05), ax, size=pic_size)
    add_pictogram(sym_bright_loom, (695, 1.05), ax, size=pic_size)
    add_pictogram(sym_dark_loom, (758, 1.05), ax, size=pic_size)
    add_pictogram(sym_bright_flash, (820, 1.05), ax, size=pic_size)
    add_pictogram(sym_dark_flash, (880, 1.05), ax, size=pic_size)


def compute_vr_regressors(vr, ca_trace, ca_sampling_rate):
    sweeps = vr.keys()
    # Generate Calcium Impulse Response Function
    tau_rise = 3  # Rise Time Constant for Calcium Impulse Response Function (seconds)
    tau_decay = 6 # Decay Time Constant for Calcium Impulse Response Function (seconds)

    _, cif = calcium_impulse_response(tau_rise=tau_rise, tau_decay=tau_decay, amplitude=1.0,
                                      sampling_rate=ca_sampling_rate, threshold=1e-5, norm=True)
    sw_motor_regs = dict()
    for sw in sweeps:
        # get motor onsets (in ca sampling rate)
        vr_onsets = vr[sw]['start_idx']
        # create a binary from motor onsets
        vr_binary = np.zeros_like(ca_trace)
        vr_binary[vr_onsets] = 1
        # create motor regressor trace
        vr_regs = create_regressors_from_binary(vr_binary, cif, delta=False, norm=False)
        sw_motor_regs[sw] = vr_regs
    return sw_motor_regs


def create_figure():
    set_plot_defaults()
    style = PlotStyle()
    base_dir = 'data/fig4'
    heatmap_data = load_heatmap_data(base_dir)
    examples_vis, sw_vis, rois_vis, rois_vis_num = get_example_traces(heatmap_data['ca'])

    # Get regressor scores
    neuron_df, p_left_right, p_left_right_frac = get_score_data(base_dir)

    # ==================================================================================================================
    # Figure
    # A4 page: The figure will be on a DinA4 page in the final pdf file
    a4_w_cm, a4_h_cm = 21.0, 29.7
    fig_width = a4_w_cm / 2.54
    fig_height = a4_h_cm / 2.54

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Center the 16x20 cm content block on the A4 page
    content_w_cm, content_h_cm = 16, 20
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

    axs = dict()

    # ------------------------------------------------------------------------------------------------------------------
    # Create axes
    # Heatmaps
    axs['stimulus'] = plt.subplot(grid[0:4, 0:70])
    axs['heatmap'] = plt.subplot(grid[5:35, 0:70])

    axs['stimulus_me'] = plt.subplot(grid[0:4, 80:150])
    axs['heatmap_me'] = plt.subplot(grid[5:35, 80:150])

    axs['with_motor_heatmap'] = plt.subplot(grid[38:68, 0:70])
    axs['no_motor_heatmap'] = plt.subplot(grid[38:68, 80:150])

    # Heatmap Colorbar
    axs['heatmap_cb'] = plt.subplot(grid[10:65, 154:156])

    # Example Traces
    axs['vis_0'] = plt.subplot(grid[90:100, 0:70])
    axs['vis_1'] = plt.subplot(grid[102:112, 0:70])
    axs['vis_2'] = plt.subplot(grid[114:124, 0:70])

    # Example Traces ZOOM
    axs['zoom_bl'] = plt.subplot(grid[90:124, 80:110])
    axs['zoom_dl'] = plt.subplot(grid[90:124, 120:150])

    axs['anatomical_map'] = plt.subplot(grid[140:200, 0:60])  # 50 x 50
    # axs['anatomical_map_colorbar'] = plt.subplot(grid[150:190, 62:64])

    axs['lr_hist'] = plt.subplot(grid[140:200, 90:150])

    # ==================================================================================================================
    # A) and B) Ca heatmap
    t_axis = heatmap_data['ca_time_axis']

    # ------------------------------------------------------------------------------------------------------------------
    # stimulus traces
    axs['stimulus'].plot(t_axis, heatmap_data['stimulus_trace'], **style.lsStimulus)
    axs['stimulus'].set_xlim([0, t_axis[-1]])
    axs['stimulus'].set_axis_off()
    axs['stimulus_me'].plot(t_axis, heatmap_data['stimulus_trace'], **style.lsStimulus)
    axs['stimulus_me'].set_xlim([0, t_axis[-1]])
    axs['stimulus_me'].set_axis_off()

    # ------------------------------------------------------------------------------------------------------------------
    # Add Stimulus Pictograms
    pic_size = 0.003
    # mt, mt, g_appears, g0, g180, g_disappears, brightL, darkL, brightF, darkF
    add_stimulus_pics(axs['stimulus'], pic_size, f'{base_dir}/images')
    add_stimulus_pics(axs['stimulus_me'], pic_size, f'{base_dir}/images')

    # ------------------------------------------------------------------------------------------------------------------
    # Create 2D array for the side-bar
    number_of_rois = heatmap_data['ca'].shape[1]
    fish_ids = create_alternating_pattern(heatmap_data['ca_labels'].iloc[0, :].to_numpy().astype('int'))
    fish_ids_bar = np.array(fish_ids).reshape(-1, 1)  # Shape: (neurons, 1)
    z_plane_bar = np.array(heatmap_data['ca_labels'].iloc[4, :]).reshape(-1, 1).astype('int')

    plot_heatmap(axs['heatmap'], heatmap_data, number_of_rois, fish_ids_bar, z_plane_bar, data_id='ca', show_me=False)
    plot_heatmap(axs['heatmap_me'], heatmap_data, number_of_rois, fish_ids_bar, z_plane_bar, data_id='vr_reg_map',
                 show_me=False)
    plot_heatmap(axs['with_motor_heatmap'], heatmap_data, number_of_rois, fish_ids_bar, z_plane_bar, data_id='ca',
                 show_me=True)
    plot_heatmap(axs['no_motor_heatmap'], heatmap_data, number_of_rois, fish_ids_bar, z_plane_bar,
                 data_id='ca_no_motor', show_me=False)

    remove_y_axis(axs['heatmap_me'], ticks=True, label=True, complete=False)
    remove_y_axis(axs['no_motor_heatmap'], ticks=True, label=True, complete=False)
    remove_x_axis(axs['heatmap'], ticks=True, label=True, complete=False)
    remove_x_axis(axs['heatmap_me'], ticks=True, label=True, complete=False)

    axs['heatmap'].text(0.01, 0.05, f'{style.deltaF}', transform=axs['heatmap'].transAxes, rotation=0, ha='left',
                        va='center', **style.txtAnnotationWhite)
    axs['heatmap_me'].text(0.01, 0.05, 'Motor regressors', transform=axs['heatmap_me'].transAxes, rotation=0, ha='left',
                           va='center', **style.txtAnnotationWhite)
    axs['with_motor_heatmap'].text(0.01, 0.05, f'{style.deltaF}', transform=axs['with_motor_heatmap'].transAxes,
                                   rotation=0, ha='left', va='center', **style.txtAnnotationWhite)
    axs['no_motor_heatmap'].text(0.01, 0.05, f'({style.deltaF}) / Motor', transform=axs['no_motor_heatmap'].transAxes,
                                 rotation=0, ha='left', va='center', **style.txtAnnotationWhite)

    axs['with_motor_heatmap'].set_xlabel('Time [s]')
    axs['no_motor_heatmap'].set_xlabel('Time [s]')
    axs['heatmap'].set_ylabel('Neuron #')
    axs['with_motor_heatmap'].set_ylabel('Neuron #')
    axs['with_motor_heatmap'].set_xticks([0, 200, 400, 600, 800])
    axs['with_motor_heatmap'].set_xlim(0, t_axis.max())
    axs['no_motor_heatmap'].set_xlim(0, t_axis.max())
    axs['no_motor_heatmap'].set_xticks([0, 200, 400, 600, 800])

    # ------------------------------------------------------------------------------------------------------------------
    # Ca heatmap colorbar
    v_min = 0
    v_max = 3
    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    cb = matplotlib.colorbar.ColorbarBase(axs['heatmap_cb'], norm=norm, cmap=style.heatmap_cmap)
    cb.ax.tick_params(labelsize=style.cbar_ticks_size)
    cb.ax.text(3.0, 0.5, f'{style.deltaF}', transform=cb.ax.transAxes, rotation=270, ha='center', va='center',
               size=style.cbar_label_size)
    cb.set_ticks([0, 1, 2, 3])

    # ==================================================================================================================
    # C) Example Traces
    y_min = -0.5
    y_max = 6
    for k in range(3):
        # Indicate Stimulus Onsets
        for onset in heatmap_data['stimulus_onsets'].iloc[0]:
            axs[f'vis_{k}'].plot([onset, onset], [y_min, y_max], **style.lsStimulusOnsetLine_solid)

        # Plot Ca Trace
        axs[f'vis_{k}'].plot(t_axis, examples_vis.iloc[:, k], **style.lsCaTrace)

        # Add Roi ID
        axs[f'vis_{k}'].text(0, 0.5, f'{rois_vis_num[k]}', ha='left', va='center', transform=axs[f'vis_{k}'].transAxes,
                             **style.txtAnnotation)

        axs[f'vis_{k}'].set_xlim(0, t_axis.max())
        axs[f'vis_{k}'].set_ylim(y_min, y_max)
        axs[f'vis_{k}'].axis('off')

    # Label area for zoom in next panel
    axs['vis_0'].axvspan(
        690,
        720,
        facecolor='gray',
        edgecolor='none',
        alpha=0.75,
        zorder=0
    )

    axs['vis_0'].axvspan(
        750,
        780,
        facecolor='gray',
        edgecolor='none',
        alpha=0.75,
        zorder=0
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Indicate Motor Event Onsets
    kk = 0
    for sw in sw_vis:
        onsets = heatmap_data['vr_events'][sw]['start_time']
        for on in onsets:
            if on < t_axis.max() - 5:
                axs[f'vis_{kk}'].plot(on, 5.5, **style.onset_marker)
        kk += 1

    # ------------------------------------------------------------------------------------------------------------------
    # Indicate purely visual responses
    ons = heatmap_data['stimulus_onsets']
    m_height = 4
    axs[f'vis_0'].plot(ons.iloc[0, [6, 8]], [[m_height]] * 2, **style.visual_marker)
    axs[f'vis_1'].plot(ons.iloc[0, [0, 2, 5, 6, 8]], [[m_height]] * 5, **style.visual_marker)
    axs[f'vis_2'].plot(ons.iloc[0, [0, 6, 8, 9]], [[m_height]] * 4, **style.visual_marker)

    # Indicate spontaneous activity
    m_height = 2.5
    axs[f'vis_0'].plot([57, 385, 724], [m_height]*3, **style.spont_act_marker)
    axs[f'vis_1'].plot([209, 848], [m_height]*2, **style.spont_act_marker)

    # Set axes ticks
    axs['vis_2'].set_xticks([0, 200, 400, 600, 800])

    # ------------------------------------------------------------------------------------------------------------------
    # Add Scale Bars
    # Time Scale Bar
    st = style.hScaleBar.copy()
    st['padding'] = 0.2
    add_scale_bar_v2(axs[f'vis_2'], size=60, label='60 s', location=(0.90, -0.15), **st)

    # origin on axis
    add_scale_bar(axs[f'vis_0'], size=2, label=f'2 {style.deltaF}', location=(0.95, 0.5), **style.vScaleBar)

    # ------------------------------------------------------------------------------------------------------------------
    # D) Zoom Bright and Dark Loom
    y_max = 5
    y_min = -0.5
    # Add pictograms
    pic_size = 0.008
    sym_bright_loom = plt.imread(f'{base_dir}/images/symbol_bright_loom.jpg')
    sym_dark_loom = plt.imread(f'{base_dir}/images/symbol_dark_loom.jpg')

    ax = axs['zoom_bl']
    ax.plot(t_axis+1, examples_vis.iloc[:, 0], color='black', lw=2.0, linestyle='-')

    # Indicate Stimulus Onsets
    for onset in heatmap_data['stimulus_onsets'].iloc[0]:
        ax.plot([onset, onset], [y_min, y_max], **style.lsStimulusOnsetLine_zoom)

    add_pictogram(sym_bright_loom, (694, y_max), ax, size=pic_size, ha='center', va='top', coord='data')

    ax.set_xlim(690, 720)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')

    ax = axs['zoom_dl']
    ax.plot(t_axis+1, examples_vis.iloc[:, 0], color='black', lw=2.0, linestyle='-')

    # Indicate Stimulus Onsets
    for onset in heatmap_data['stimulus_onsets'].iloc[0]:
        ax.plot([onset, onset], [y_min, y_max], **style.lsStimulusOnsetLine_zoom)

    # Indicate Motor Event Onsets
    sw = 'sw_03'
    onsets = heatmap_data['vr_events'][sw]['start_time']
    swim_onset = onsets[28]
    swim_offset = onsets[32]
    ax.plot([swim_onset, swim_offset], [4.2, 4.2], color='tab:blue', lw=3)
    add_pictogram(sym_dark_loom, (756, y_max), ax, size=pic_size, ha='center', va='top', coord='data')

    ax.set_xlim(750, 780)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')

    # Time Scale Bar
    add_scale_bar(ax, size=1, label=f' 1 {style.deltaF}', location=(0.75, 0.7), **style.vScaleBar)
    add_scale_bar(ax, size=5, label=f'5 s', location=(0.75, 0.7), **style.hScaleBar)

    # ------------------------------------------------------------------------------------------------------------------
    # E) Anatomical map of purely visual stimulus-type counts
    plot_visual_anatomy_binary(
        axs['anatomical_map'],
        neuron_df
    )
    hide_axis_spines(axs['anatomical_map'])

    # ------------------------------------------------------------------------------------------------------------------
    # F) Left-right histogram
    plot_visual_left_right_hist(axs['lr_hist'], neuron_df, p_left_right_frac, style)
    hide_axis_spines(axs['lr_hist'])

    # ------------------------------------------------------------------------------------------------------------------
    # Add Sub-Figure Caps
    # from plotting_utils import set_caption_labels
    # set_caption_labels(fig)
    # plt.show()
    # exit()

    cap_pos = {
        'A': (np.float64(0.03), np.float64(0.95)),
        'B': (np.float64(0.50), np.float64(0.95)),
        'C': (np.float64(0.03), np.float64(0.58)),
        'D': (np.float64(0.50), np.float64(0.58)),
        'E': (np.float64(0.03), np.float64(0.33)),
        'F': (np.float64(0.50), np.float64(0.33))
    }

    # Redefine position to DinA4 page
    def content_xy(fx, fy):
        return left + fx * (right - left), bottom + fy * (top - bottom)

    for cap, (fx, fy) in cap_pos.items():
        x, y = content_xy(fx, fy)
        fig.text(x, y, cap, **style.cap_style)

    # ------------------------------------------------------------------------------------------------------------------
    # Store figures
    fig_dir = f"figs"
    # bbox_inches=None, otherwise figure will get cropped
    matplotlib.rcParams["savefig.bbox"] = None
    fig_nr = 4

    # Add figure name to pdf
    fig.text(style.figure_label_pos[0], style.figure_label_pos[1], f'Figure {fig_nr}', **style.figure_label)

    plt.savefig(f'{fig_dir}/04_figure_{fig_nr}.pdf', dpi=600)
    plt.close(fig)
    print(f'\n=== STORED FIGURE {fig_nr} TO HDD ====\n')


def main():
    create_figure()


if __name__ == "__main__":
    main()

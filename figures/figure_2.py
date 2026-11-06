"""
Assemble and export Figure 2 for the manuscript:
A dopaminergic visuomotor pathway to the zebrafish optic tectum
Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann
iScience, Cell Press 2026

Figure 2: Functional imaging of PrDA neurons

"""
# Imports
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from plotting_utils import add_scale_bar, add_row_label, add_pictogram
from plotting_style import PlotStyle, set_plot_defaults
from utils import calcium_impulse_response, create_regressors_from_binary, load_hdf5_as_dict, merge_events_closing


def plot_figure(data, rois):
    set_plot_defaults()
    style = PlotStyle()

    # Load Images
    base_dir = 'data/fig2/images'
    setup_img = plt.imread(f'{base_dir}/stimulus_setup.png')
    fov_img = plt.imread(f'{base_dir}/fov2.png')
    sym_mt = plt.imread(f'{base_dir}/symbol_mt.jpg')
    sym_dark_loom = plt.imread(f'{base_dir}/symbol_dark_loom.jpg')
    sym_bright_loom = plt.imread(f'{base_dir}/symbol_bright_loom.jpg')
    sym_grating_appears = plt.imread(f'{base_dir}/symbol_grating_appears.jpg')
    sym_grating_0 = plt.imread(f'{base_dir}/symbol_grating_0.jpg')
    sym_grating_180 = plt.imread(f'{base_dir}/symbol_grating_180.jpg')
    sym_dark_flash = plt.imread(f'{base_dir}/symbol_dark_flash.jpg')
    sym_bright_flash = plt.imread(f'{base_dir}/symbol_bright_flash.jpg')

    circuit_schematic = plt.imread(f'{base_dir}/circuit_schematic.png')

    # Organize data
    stimulus_trace = data['stimulus_trace']
    stimulus_binary = data['stimulus_binary']
    vr_trace = data['vr_recording']
    vr_events = data['vr_events']
    ca_traces = data['ca_traces']
    vr_fr = float(data['vr_sampling_rate'].values[0][0])
    ca_fr = float(data['ca_sampling_rate'].values[0][0])

    # Get spontaneous and stimulus elicited motor events
    idx_sp = vr_events['stimulus'] == 'spontaneous'
    vr_sp = vr_events[idx_sp]
    vr_st = vr_events[~idx_sp]

    # Create Time Axis
    vr_time_axis = np.linspace(0, vr_trace.shape[0] / vr_fr, vr_trace.shape[0])
    ca_time_axis = np.linspace(0, ca_traces.shape[0] / ca_fr, ca_traces.shape[0])

    # Generate Calcium Impulse Response Function
    tau_rise = 3
    tau_decay = 6

    _, cif = calcium_impulse_response(tau_rise=tau_rise, tau_decay=tau_decay, amplitude=1.0,
                                      sampling_rate=ca_fr, threshold=1e-5, norm=True)

    # Get Stimulus Regressor Traces
    stimulus_regs = dict()
    for s_type in stimulus_binary:
        stimulus_regs[s_type] = create_regressors_from_binary(stimulus_binary[s_type], cif, delta=True, norm=True)

    # Get Motor Regressor Traces
    vr_event_binary = np.zeros_like(ca_time_axis)
    vr_event_binary[vr_events['start_idx']] = 1
    vr_event_binary = merge_events_closing(data['vr_binary'], int(8 * ca_fr))

    vr_reg = create_regressors_from_binary(vr_event_binary, cif, delta=True, norm=False)

    # ==================================================================================================================
    # Figure
    # A4 page: The figure will be on a DinA4 page in the final pdf file
    a4_w_cm, a4_h_cm = 21.0, 29.7
    fig_width = a4_w_cm / 2.54
    fig_height = a4_h_cm / 2.54

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Center the wxh cm content block on the A4 page
    content_w_cm, content_h_cm = 16, 21
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

    axs = dict()
    axs_img = dict()


    # ==================================================================================================================
    # Create axes
    axs_img['setup'] = plt.subplot(grid[0:30, 0:55])
    axs_img['stimulus_pics'] = plt.subplot(grid[0:30, 55:80])
    axs_img['fov'] = plt.subplot(grid[0:35, 80:130])
    axs_img['circuit_schematic'] = plt.subplot(grid[0:35, 130:160])  # 1.2x, 1152 x 1368

    axs['stimulus'] = plt.subplot(grid[45:50, 0:160])
    axs['ventral_root'] = plt.subplot(grid[55:75, 0:160])
    axs['ventral_root_reg'] = plt.subplot(grid[80:90, 0:160])
    axs['stimulus_reg'] = plt.subplot(grid[95:105, 0:160])
    axs['ca_trace_0'] = plt.subplot(grid[110:130, 0:160])
    axs['ca_trace_1'] = plt.subplot(grid[130:150, 0:160])
    axs['ca_trace_2'] = plt.subplot(grid[150:170, 0:160])
    axs['ca_trace_3'] = plt.subplot(grid[170:190, 0:160])
    axs['ca_trace_4'] = plt.subplot(grid[190:210, 0:160])

    # PLOTTING =========================================================================================================
    # A) and B) Setup and FOV Images
    axs_img['setup'].imshow(setup_img, interpolation=None)
    axs_img['fov'].imshow(fov_img)
    axs_img['setup'].axis('off')
    axs_img['fov'].axis('off')

    # Add Stimulus Description
    pic_size = 0.004
    add_pictogram(sym_mt, (0, 0.91), axs_img['stimulus_pics'], size=pic_size)
    add_pictogram(sym_grating_appears, (0, 0.78), axs_img['stimulus_pics'], size=pic_size)
    add_pictogram(sym_grating_0, (0, 0.65), axs_img['stimulus_pics'], size=pic_size)
    add_pictogram(sym_grating_180, (0, 0.52), axs_img['stimulus_pics'], size=pic_size)
    add_pictogram(sym_bright_loom, (0, 0.39), axs_img['stimulus_pics'], size=pic_size)
    add_pictogram(sym_dark_loom, (0, 0.26), axs_img['stimulus_pics'], size=pic_size)
    add_pictogram(sym_bright_flash, (0, 0.13), axs_img['stimulus_pics'], size=pic_size)
    add_pictogram(sym_dark_flash, (0, 0), axs_img['stimulus_pics'], size=pic_size)
    axs_img['stimulus_pics'].axis('off')

    axs_img['stimulus_pics'].text(0.1, 0.91+0.02, 'Moving Target', transform=axs_img['stimulus_pics'].transAxes, ha='left', va='bottom', **style.txtStimulusPics)
    axs_img['stimulus_pics'].text(0.1, 0.78+0.02, 'Grating appears/disappears', transform=axs_img['stimulus_pics'].transAxes, ha='left', va='bottom', **style.txtStimulusPics)
    axs_img['stimulus_pics'].text(0.1, 0.65+0.02, 'Grating forward', transform=axs_img['stimulus_pics'].transAxes, ha='left', va='bottom', **style.txtStimulusPics)
    axs_img['stimulus_pics'].text(0.1, 0.52+0.02, 'Grating backward', transform=axs_img['stimulus_pics'].transAxes, ha='left', va='bottom', **style.txtStimulusPics)
    axs_img['stimulus_pics'].text(0.1, 0.39+0.02, 'Bright Loom (white disc)', transform=axs_img['stimulus_pics'].transAxes, ha='left', va='bottom', **style.txtStimulusPics)
    axs_img['stimulus_pics'].text(0.1, 0.26+0.02, 'Dark Loom (dark disc)', transform=axs_img['stimulus_pics'].transAxes, ha='left', va='bottom', **style.txtStimulusPics)
    axs_img['stimulus_pics'].text(0.1, 0.13+0.02, 'Bright Flash (white screen)', transform=axs_img['stimulus_pics'].transAxes, ha='left', va='bottom', **style.txtStimulusPics)
    axs_img['stimulus_pics'].text(0.1, 0.00+0.02, 'Dark Flash (dark screen)', transform=axs_img['stimulus_pics'].transAxes, ha='left', va='bottom', **style.txtStimulusPics)

    # ------------------------------------------------------------------------------------------------------------------
    # C) Circuit Schematic
    ax = axs_img['circuit_schematic']
    ax.imshow(circuit_schematic, interpolation=None)
    ax.axis('off')

    # ------------------------------------------------------------------------------------------------------------------
    # D) Example recording
    # Add Stimulus Pictograms
    pic_size = 0.005
    # mt, mt, g_appears, g0, g180, g_disappears, brightL, darkL, brightF, darkF
    add_pictogram(sym_mt, (302, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_mt, (370, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_grating_appears, (434, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_grating_0, (500, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_grating_180, (570, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_grating_appears, (634, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_bright_loom, (695, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_dark_loom, (758, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_bright_flash, (820, 1.05), axs['stimulus'], size=pic_size)
    add_pictogram(sym_dark_flash, (880, 1.05), axs['stimulus'], size=pic_size)

    # ------------------------------------------------------------------------------------------------------------------
    # STIMULUS TRACE
    axs['stimulus'].plot(ca_time_axis, stimulus_trace, **style.lsStimulus)

    # ------------------------------------------------------------------------------------------------------------------
    # VENTRAL ROOT TRACE
    axs['ventral_root'].plot(vr_time_axis, vr_trace, **style.lsVrTrace)

    # ------------------------------------------------------------------------------------------------------------------
    # Indicate Motor Event Onsets
    for onset in vr_sp['start_time']:
        axs['ventral_root'].plot(onset, 15, **style.onset_marker)

    for onset in vr_st['start_time']:
        axs['ventral_root'].plot(onset, 15, **style.onset_marker_st)

    # Indicate Artefacts
    artefacts = [241, 412, 491, 756]
    for a in artefacts:
        axs['ventral_root'].plot(a, 10, **style.artefact_marker)

    # ------------------------------------------------------------------------------------------------------------------
    # VENTRAL ROOT REG
    axs['ventral_root_reg'].plot(ca_time_axis, vr_reg, **style.lsVrReg)
    axs['ventral_root_reg'].set_ylim(-0.1, 1.1)

    # ------------------------------------------------------------------------------------------------------------------
    # STIMULUS REG
    all_regs = np.array([sum(values) for values in zip(*stimulus_regs.values())])
    # mt_reg = stimulus_regs['moving_target_01'] + stimulus_regs['moving_target_02']
    bl_reg = stimulus_regs['bright_loom']
    axs['stimulus_reg'].plot(ca_time_axis, all_regs, **style.lsStimulusReg_fading)
    axs['stimulus_reg'].plot(ca_time_axis, bl_reg, **style.lsStimulusReg_highlight)
    axs['stimulus_reg'].set_ylim(-0.1, 1.1)

    # ------------------------------------------------------------------------------------------------------------------
    # CA IMAGING TRACES
    for k_roi in range(5):
        axs[f'ca_trace_{k_roi}'].plot(ca_time_axis, ca_traces[rois[k_roi]], **style.lsCaTrace)
        axs[f'ca_trace_{k_roi}'].fill_between(ca_time_axis, all_regs * 3, y2=0, **style.fsStimulusReg)
        axs[f'ca_trace_{k_roi}'].fill_between(ca_time_axis, vr_reg * 3, y2=0, **style.fsVrReg)
        axs[f'ca_trace_{k_roi}'].fill_between(ca_time_axis, (vr_reg * all_regs) * 3, y2=0, **style.fsOverlap)
        axs[f'ca_trace_{k_roi}'].fill_between(ca_time_axis, y1=-0.1, y2=0.001, edgecolor='white', facecolor='white')
        axs[f'ca_trace_{k_roi}'].set_ylim(-1, 8)
        add_row_label(axs[f'ca_trace_{k_roi}'], 0, -0.1, f'{style.deltaF}', style.txtCaTraceLabel)

    # ------------------------------------------------------------------------------------------------------------------
    # Row Labels
    x_pos = 0
    add_row_label(axs['stimulus'], x_pos, -0.5, 'Stimulus', style.txtStimulusLabel)
    add_row_label(axs['ventral_root'], x_pos, -0.05, 'MNR', style.txtVRLabel)
    add_row_label(axs['ventral_root_reg'], x_pos, -0.3, 'Motor regressor', style.txtMotorLabel)
    add_row_label(axs['stimulus_reg'], x_pos, -0.3, 'Stimulus regressor', style.txtStimulusLabel)
    # ------------------------------------------------------------------------------------------------------------------
    # Axes
    # X Limit
    x_lim_min = 60  # secs
    x_lim_max = 960
    for ax in axs:
        axs[ax].set_xlim(x_lim_min, x_lim_max)
        # Remove Axes
        axs[ax].axis('off')

    # Y Limit
    axs['ventral_root'].set_ylim(-12, 20)

    # ------------------------------------------------------------------------------------------------------------------
    # Add Scale Bars
    # Time Scale Bar
    add_scale_bar(
        axs['ca_trace_4'], size=60, label='60 s', location=(0.90, 0.03), orientation='horizontal', color='black',
        linewidth=3, fontsize=10, padding=0.08)

    # Delta F over F Scale Bar
    add_scale_bar(
        axs['ca_trace_0'], size=2, label=f'2 {style.deltaF}', location=(0.90, 0.50), orientation='vertical', color='black',
        linewidth=3, fontsize=10, padding=0.02)

    add_scale_bar(
        axs['ventral_root_reg'], size=0.5, label=f'0.5 {style.deltaF}', location=(0.90, 0.40), orientation='vertical', color='black',
        linewidth=3, fontsize=10, padding=0.02)

    # Ventral Root mV Scale Bar
    add_scale_bar(
        axs['ventral_root'], size=5, label='5 mV', location=(0.95, 0.7), orientation='vertical', color='black',
        linewidth=3, fontsize=10, padding=0.02)

    # ------------------------------------------------------------------------------------------------------------------
    # Add Sub-Figure Caps
    cap_x = [0, 0, 0, 0, 0, 0]
    cap_y = [1.1, 1.1, 1.1, 1.2, 1.1, 1.1]
    subfig_caps_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

    for i, ax in enumerate([axs_img['setup'], axs_img['fov'], axs_img['circuit_schematic'], axs['stimulus']]):
        ax.text(
            cap_x[i],
            cap_y[i],
            subfig_caps_labels[i],
            transform=ax.transAxes,
            **style.cap_style
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Store Figure to HDD
    matplotlib.rcParams["savefig.bbox"] = None
    fig_dir = f"figs"
    fig_nr = 2

    # Add figure name to pdf
    fig.text(style.figure_label_pos[0], style.figure_label_pos[1], f'Figure {fig_nr}', **style.figure_label)

    plt.savefig(f'{fig_dir}/02_figure_{fig_nr}.pdf', dpi=600)

    plt.close(fig)
    print(f'\n=== STORED FIGURE {fig_nr} TO HDD ====\n')


def main():
    # Load data
    data_dir = 'data/fig2/example_data.h5'
    data = load_hdf5_as_dict(data_dir)

    # Create Figure
    plot_figure(data, rois=['457', '461', '462', '458', '460'])


if __name__ == '__main__':
    main()

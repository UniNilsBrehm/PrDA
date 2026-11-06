import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from collections import OrderedDict


def set_plot_defaults() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 1,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1,
        "lines.color": "k",
        "lines.markersize": 2,
        "errorbar.capsize": 0,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "pcolor.shading": "auto",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        "savefig.bbox": None
    })


def create_color_map(mode: str = "grey") -> LinearSegmentedColormap:
    if mode == "green":
        colors = [(0, 0, 0), (0, 1, 0)]
    elif mode == "red":
        colors = [(0, 0, 0), (1, 0, 0)]
    elif mode == "magenta":
        colors = [(0, 0, 0), (1, 0, 1)]
    elif mode == "corr":
        colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]
    else:
        colors = [(0, 0, 0), (1, 1, 1)]

    return LinearSegmentedColormap.from_list(f"custom_{mode}", colors, N=512)


class PlotStyle:
    def __init__(self):
        self.line_styles = OrderedDict(
            [('solid', (0, ())),
             ('loosely dotted', (0, (1, 10))),
             ('dotted', (0, (1, 5))),
             ('densely dotted', (0, (1, 1))),

             ('loosely dashed', (0, (5, 10))),
             ('dashed', (0, (5, 5))),
             ('densely dashed', (0, (5, 1))),

             ('loosely dashdotted', (0, (3, 10, 1, 10))),
             ('dashdotted', (0, (3, 5, 1, 5))),
             ('densely dashdotted', (0, (3, 1, 1, 1))),

             ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
             ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
             ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

        # default maker size
        self.marker_size = 3
        self.scatter_size = 8

        self.deltaF = r'$\Delta F / F$'

        self.signal_color = 'black'
        self.motor_color = 'tab:blue'
        self.motor_color_st = 'violet'
        self.stimulus_color = 'tab:green'
        self.sub_color = 'lightcoral'
        self.additive_color = 'black'
        self.supra_color = 'seagreen'

        # Figure 3
        self.fig_3_mark_style = dict(color='tab:blue', linestyle='-', linewidth=1, alpha=1)
        self.fig_3_mark_style_dashed = dict(color='tab:blue', linestyle='--', linewidth=1, alpha=1)
        self.fig_3_hist_events_color = "black"
        self.fig_3_hist_prob_color = "black"
        self.fig_3_example01_color = "magenta"
        self.fig_3_example02_color = "cyan"
        self.fig_3_example_marker_size = 7

        # Line Styles
        # Fig 1 Example
        self.lsStimulusExample = dict(color=self.stimulus_color, lw=2.0, linestyle='-')
        self.lsCaTraceExample = dict(color="black", lw=2.0, linestyle='-')
        self.lsVrTraceExample = dict(color='gray', lw=1.5, linestyle='-')

        self.lsStimulus = dict(color=self.stimulus_color, lw=1.0, linestyle='-')
        self.lsCaTrace = dict(color='black', lw=1.0, linestyle='-')
        self.lsCaTrace_shifted = dict(color='black', lw=1.0, linestyle=':')
        self.lsVrTrace = dict(color='gray', lw=1.0, linestyle='-')
        self.lsVrReg = dict(color=self.motor_color, lw=1.0, linestyle='-')
        self.lsVrReg_spontaneous = dict(color='tab:orange', lw=1.0, linestyle='-')
        self.lsStimulusReg = dict(color=self.stimulus_color, lw=1.5, linestyle='-')
        self.lsStimulusReg_highlight = dict(color=self.stimulus_color, lw=1.5, linestyle=self.line_styles['densely dotted'])
        self.lsStimulusReg_fading = dict(color=self.stimulus_color, lw=1.0, linestyle='-', alpha=0.2)
        self.onset_marker = dict(marker="v", markersize=self.marker_size, markerfacecolor=self.motor_color, markeredgecolor=self.motor_color, linestyle="None")
        self.onset_marker_st = dict(marker="v", markersize=self.marker_size, markerfacecolor=self.motor_color_st, markeredgecolor=self.motor_color_st, linestyle="None")
        self.artefact_marker = dict(marker="s", markersize=self.marker_size, markerfacecolor='black', markeredgecolor='black', linestyle="None")
        self.visual_marker = dict(marker="v", markersize=self.marker_size, markerfacecolor='tab:red', markeredgecolor='tab:red', linestyle="None")
        self.spont_act_marker = dict(marker="v", markersize=self.marker_size-1, markerfacecolor='tab:olive', markeredgecolor='tab:olive', linestyle="None")
        self.onset_marker_large = dict(marker="v", markersize=self.marker_size+2, markerfacecolor=self.motor_color, markeredgecolor=self.motor_color, linestyle="None")

        self.onset_marker_stimulus = dict(marker="v", markersize=3, markerfacecolor=self.stimulus_color, markeredgecolor=self.stimulus_color, linestyle="None")
        self.lsRegressionLine = dict(color='tab:red', lw=1.0, linestyle='--', alpha=0.75)
        self.lsMidLine = dict(color='tab:red', lw=1.0, linestyle='--', alpha=1)
        self.lsThresholdLine = dict(color='tab:red', lw=1.0, linestyle=':', alpha=1)
        self.lsThresholdLineScatter = dict(color='black', lw=1.0, ls=':', alpha=1)
        self.lsStimulusOnsetLine = dict(color=self.stimulus_color, lw=1, linestyle='--')
        self.lsStimulusOnsetLine_solid = dict(color=self.stimulus_color, lw=0.5, linestyle='-')
        self.lsStimulusOnsetLine_zoom = dict(color=self.stimulus_color, lw=1, linestyle='-')

        self.lsDS_Zero = dict(color='black', linestyle=(0, (3, 1)), linewidth=1.5, dash_capstyle='butt')

        # Heatmaps
        # Black and white colormap
        self.heatmap_cmap = 'afmhot'
        # self.heatmap_cmap = 'magma'
        self.fish_ids_cmap = ListedColormap(['black', 'white'])
        self.z_plane_cmap = ListedColormap(['red', 'blue', 'green', 'magenta', 'cyan', 'orange'])
        self.motor_onsets = dict(color=self.motor_color, lw=0.5, alpha=0.75)

        # Face Styles
        self.fsStimulusReg = dict(edgecolor=self.stimulus_color, facecolor=self.stimulus_color, alpha=1)
        self.fsVrReg = dict(edgecolor=self.motor_color, facecolor=self.motor_color, alpha=1)
        self.fsOverlap = dict(edgecolor='tab:orange', facecolor='tab:orange', alpha=1)

        # Scatter Styles
        self.scScoring = dict(marker="o", s=self.scatter_size, alpha=1)
        self.scCorr = dict(marker="o", c='gray', s=self.scatter_size, alpha=1)
        self.scCorr_transparent = dict(marker="o", c='lightgray', s=self.scatter_size, alpha=1)
        self.scCorr_vis = dict(marker="o", s=self.scatter_size, c=self.stimulus_color, alpha=1)
        self.scCorr_mixed = dict(marker="o", s=self.scatter_size, c='tab:orange', alpha=1)
        self.anatomical_cmap = 'viridis'
        self.scAnatomical_scatter = dict(cmap=self.anatomical_cmap, marker="o", s=20, alpha=1.0)
        self.scAnatomical_visual = dict(marker="o", s=20, alpha=1.0)

        # Error bar styles
        self.error_bar = dict(fmt="o", capsize=0, linewidth=2, markersize=5)

        # Histograms
        self.histogram_style = dict(edgecolor='black', facecolor='gray', histtype='bar', linewidth=0.5)
        self.histogram_vis_style = dict(edgecolor=self.stimulus_color, facecolor=self.stimulus_color, histtype='bar')
        self.histogram_vis_motor_style = dict(edgecolor="tab:orange", facecolor="tab:orange", histtype='bar')
        self.histogram_style_variation1 = dict(edgecolor='black', facecolor='tab:olive', histtype='bar')
        self.histogram_style_variation2 = dict(edgecolor='black', facecolor='tab:purple', histtype='bar')

        # Text Styles
        self.sig_star_size = 12
        self.ns_dot_size = 6
        self.txtTimePoints = dict(color='white', fontsize=8)
        self.txtN = dict(color='black', fontsize=8)
        self.txtLabel = dict(color='black', fontsize=10)
        self.txtCaps = dict(color='black', fontsize=16)
        self.txtStimulusPics = dict(color='black', fontsize=6)

        self.txtStimulusLabel = dict(color=self.stimulus_color, fontsize=8)
        self.txtVRLabel = dict(color='gray', fontsize=8)
        self.txtMotorLabel = dict(color=self.motor_color, fontsize=8)
        self.txtCaTraceLabel = dict(color='black', fontsize=8)
        self.txtAnnotation = dict(color='black', fontsize=7)
        # self.txtAnnotationBlue = dict(color='tab:green', fontsize=7)
        self.txtAnnotationGreen = dict(color='tab:green', fontsize=7)
        self.txtAnnotationWhite = dict(color='white', fontsize=8)

        self.txtLabelGPF = dict(color='limegreen', size=8)
        self.txtLabelRFP = dict(color='magenta', size=8)

        # SI Index text
        self.txtSI_mean = dict(ha='center', va='center', fontsize=6, color='black')
        self.txtSI = dict(ha='center', va='center', fontsize=8, color='black')

        # Color maps and color bars
        self.cbar_ticks_size = 8
        self.cbar_label_size = 8
        self.cmap_default = 'afmhot'
        # self.cmap_default = 'magma'
        self.cmap_error_landscape = 'afmhot'
        self.cmap_time = 'viridis'
        self.cmap_stimulus = create_color_map(mode='magenta')
        # Define discrete colors for each class
        # (0=mixed, 1=stimulus, 2=no response)
        self.cmap_classes = ListedColormap(["tab:orange", self.stimulus_color, "gray"])
        self.cmap_anatomy_visual = 'tab10'
        self.anatomy_visual_zero_color = 'black'
        self.left_color = 'lightgray'
        self.right_color = 'dimgray'

        # Separation Lines
        self.sep_line = dict(color='r', linestyle='-',  lw=1.0)

        # Scale bars
        self.hScaleBar = dict(orientation='horizontal', color='black', linewidth=3, fontsize=6, padding=0.08)
        self.vScaleBar = dict(orientation='vertical', color='black', linewidth=3, fontsize=6, padding=0.02)

        # Sub Fig Cap Text Size
        self.cap_style = dict(size=12, color='black', weight='bold')
        # self.sub_fig_cap_text_size = 12
        # self.sub_fig_text_color = 'black'
        self.sub_fig_cap_upper_case = True
        self.subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

        self.figure_label_pos = (0.1, 0.05)
        self.figure_label = dict(size=16, color="black", fontfamily='Arial', weight='bold')
        # Colors for Pie Plot
        self.pie_colors = [
            "#1f77b4",  # blue
            "#aec7e8",  # light blue
            "#7f7f7f",  # gray
            "#2ca02c"  # green
        ]
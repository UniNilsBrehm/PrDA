"""
Assemble and export Figure 1 for the manuscript:
A dopaminergic visuomotor pathway to the zebrafish optic tectum
Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann
iScience, Cell Press 2026

Figure 1: Anatomical organization and in situ HCR of marker genes in PrDA neurons

"""

# Imports
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from plotting_utils import hide_axis_spines, add_ap_axis, add_label
from plotting_style import PlotStyle, set_plot_defaults


def _parse_value(s, decimals):
    """Convert a possibly space-grouped numeric string to a float.

    The exporter sometimes writes numbers as space-separated digit
    groups instead of decimals (e.g. "58 904 761" instead of
    "58.904761"). This reinserts the decimal point when that happens.
    """
    if pd.isna(s):
        return np.nan
    s = str(s)
    if " " in s:
        digits = s.replace(" ", "")
        return float(digits) / (10 ** decimals)
    return float(s)


def plot_stratification_profile(raw, ax, n_points=101):
    """
    Plot mean +/- SEM normalized stratification profiles (thGFP & ath5:RFP)
    across larvae onto a given matplotlib ax.

    Parameters
    ----------
    raw: str
        Path to the stratification profile CSV. Expected columns:
        "<larva>_distance", "<larva>_thgfp", "<larva>_ath5rfp" for each
        larva (any number of larvae, detected automatically).
    ax : matplotlib.axes.Axes
        Axes to draw on.
    n_points : int
        Number of points on the common 0-100 % depth axis.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # auto-detect larva names from column prefixes
    larvae = sorted({c.rsplit("_", 1)[0] for c in raw.columns if c.endswith("_distance")})

    x_common = np.linspace(0, 100, n_points)
    thgfp_interp = np.zeros((len(larvae), n_points))
    ath5rfp_interp = np.zeros((len(larvae), n_points))

    for i, larva in enumerate(larvae):
        dist = raw[f"{larva}_distance"].apply(lambda s: _parse_value(s, 5))
        thgfp = raw[f"{larva}_thgfp"].apply(lambda s: _parse_value(s, 6))
        ath5rfp = raw[f"{larva}_ath5rfp"].apply(lambda s: _parse_value(s, 6))

        df = pd.DataFrame({"distance": dist, "thgfp": thgfp, "ath5rfp": ath5rfp}).dropna()

        depth_pct = df["distance"] / df["distance"].max() * 100
        thgfp_norm = (df["thgfp"] - df["thgfp"].min()) / (df["thgfp"].max() - df["thgfp"].min())
        ath5rfp_norm = (df["ath5rfp"] - df["ath5rfp"].min()) / (df["ath5rfp"].max() - df["ath5rfp"].min())

        thgfp_interp[i] = np.interp(x_common, depth_pct, thgfp_norm)
        ath5rfp_interp[i] = np.interp(x_common, depth_pct, ath5rfp_norm)

    n = len(larvae)
    thgfp_mean = thgfp_interp.mean(axis=0)
    thgfp_sem = thgfp_interp.std(axis=0, ddof=1) / np.sqrt(n)

    ath5rfp_mean = ath5rfp_interp.mean(axis=0)
    ath5rfp_sem = ath5rfp_interp.std(axis=0, ddof=1) / np.sqrt(n)

    green = "#2ca02c"
    magenta = "#d62789"

    ax.plot(x_common, thgfp_mean, color=green, lw=2, label="PrDAtep:GFP")
    ax.fill_between(x_common, thgfp_mean - thgfp_sem, thgfp_mean + thgfp_sem,
                    color=green, alpha=0.25, linewidth=0)

    ax.plot(x_common, ath5rfp_mean, color=magenta, lw=2, label="RGC Axons:RFP")
    ax.fill_between(x_common, ath5rfp_mean - ath5rfp_sem, ath5rfp_mean + ath5rfp_sem,
                    color=magenta, alpha=0.25, linewidth=0)

    ax.axvline(x=15, color="black", linestyle="--", linewidth=1)
    ax.axvline(x=30, color="black", linestyle="--", linewidth=1)
    ax.axvline(x=90, color="black", linestyle="--", linewidth=1)
    ax.axvline(x=100, color="black", linestyle="--", linewidth=1)

    ax.spines[["top", "right"]].set_visible(False)

    return ax


def show_image(ax, img, anchor="NW", rasterized=False, aspect="equal"):
    ax.imshow(img, interpolation="none", aspect=aspect, rasterized=rasterized)  # no distortion
    ax.set_axis_off()
    # ax.set_anchor("C")  # center if there is letterboxing
    ax.set_anchor(anchor)

    if aspect == 'equal':
        # Make the axes box respect the image aspect (prevents the "fill slot by stretching" behavior)
        h, w = img.shape[:2]
        ax.set_box_aspect(h / w)  # height/width


def plot_figure():
    """
    Assemble, annotate, and export the final Figure 1 layout.
    """
    set_plot_defaults()
    style = PlotStyle()

    # ==================================================================================================================
    # Load data
    hcr_data_th_dat = pd.read_csv('data/fig1/th_dat_vmat2.csv')
    hcr_data_gad_vglut = pd.read_csv(f'data/fig1/gad1_vglut2.csv')
    hcr_data_drd = pd.read_csv(f'data/fig1/drd1_drd2.csv')

    hcr_data = pd.concat([hcr_data_th_dat, hcr_data_gad_vglut, hcr_data_drd]).reset_index(drop=True)
    line_plot_data = pd.read_csv('data/fig1/stratification_profile.csv', dtype=str)

    # Load Images
    base_img = 'data/fig1/images'
    brain_schematic = plt.imread(f'{base_img}/brain_schematic.png')
    af9 = plt.imread(f'{base_img}/af9_projections_merged.png')
    confocal_overview = plt.imread(f'{base_img}/confocal_overview.png')
    confocal_zoom = plt.imread(f'{base_img}/confocal_zoom.png')
    confocal_linescan = plt.imread(f'{base_img}/confocal_line_scan.png')
    hcr = plt.imread(f'{base_img}/hcr_combined_1_row.png')


    # ==================================================================================================================
    # Figure
    # A4 page: The figure will be on a DinA4 page in the final pdf file
    a4_w_cm, a4_h_cm = 21.0, 29.7
    fig_width = a4_w_cm / 2.54
    fig_height = a4_h_cm / 2.54

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Center the figure content block on the A4 page
    content_w_cm, content_h_cm = 16, 16
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

    axs_img = dict()

    # ------------------------------------------------------------------------------------------------------------------
    # Create Axes
    axs_img['brain_schematic'] = plt.subplot(grid[0:40, 0:50])  # 2.2x , 50 x 38 mm  (B x H)
    axs_img['confocal_overview'] = plt.subplot(grid[0:50, 55:105])  # 1.3x,  1772 x 1330 px
    axs_img['confocal_zoom'] = plt.subplot(grid[0:50, 110:160])  # 1.3x, 1299 x 976 px

    axs_img['af9'] = plt.subplot(grid[50:88, 0:29])  # Image: 945 x 709
    axs_img['line_scan'] = plt.subplot(grid[50:88, 33:73])  # 1x , 1064 x 1064 px (BxH)
    axs_img['line_plot'] = plt.subplot(grid[50:78, 88:160])

    axs_img['hcr'] = plt.subplot(grid[100:130, 0:88])  # 60 x 177 mm
    axs_img['hcr_per'] = plt.subplot(grid[100:125, 105:160])


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ------------------------------------------------------------------------------------------------------------------
    # A) Brain schematic
    show_image(axs_img['brain_schematic'], brain_schematic)

    # ------------------------------------------------------------------------------------------------------------------
    # B) and C) Confocal overview and zoom in
    show_image(axs_img['confocal_overview'], confocal_overview, rasterized=True)
    show_image(axs_img['confocal_zoom'], confocal_zoom, rasterized=True)

    # Add genotype labels
    add_label(axs_img['confocal_overview'], x_pos=0.02, y_pos=0.83, label='th;GFP', **style.txtLabelGPF)
    add_label(axs_img['confocal_overview'], x_pos=0.02, y_pos=0.93, label='atoh7;RFP', **style.txtLabelRFP)

    add_label(axs_img['confocal_zoom'], x_pos=0.02, y_pos=0.83, label='th;GFP', **style.txtLabelGPF)
    add_label(axs_img['confocal_zoom'], x_pos=0.02, y_pos=0.93, label='atoh7;RFP', **style.txtLabelRFP)

    add_ap_axis(axs_img['confocal_zoom'], start=(0.05, 0.05), end=(0.05, 0.35), color='white', labels=("P", "A"))

    # Add arrow heads
    axs_img['confocal_overview'].plot(
        0.43, 0.75,  # position
        marker='v',  # triangle up (v, ^, <, > for other directions)
        color='white',
        markersize=6,
        transform=axs_img['confocal_overview'].transAxes
    )
    axs_img['confocal_overview'].plot(
        0.57, 0.75,  # position
        marker='v',  # triangle up (v, ^, <, > for other directions)
        color='white',
        markersize=6,
        transform=axs_img['confocal_overview'].transAxes
    )

    # ------------------------------------------------------------------------------------------------------------------
    # D) AF 9
    ax = axs_img['af9']
    ax.imshow(af9, aspect='equal', interpolation=None, rasterized=True)

    # add A-P axis
    add_ap_axis(ax, start=(0.07, 0.05), end=(0.07, 0.35), color='white', labels=("P", "A"))

    # add genotype labels
    add_label(ax, x_pos=0.02, y_pos=0.95, label='PrDA dendrites', **style.txtLabelGPF)
    add_label(ax, x_pos=0.02, y_pos=0.88, label='AF9', **style.txtLabelRFP)

    ax.axis('off')

    # ------------------------------------------------------------------------------------------------------------------
    # E) Confocal Line Scan Overview
    ax = axs_img['line_scan']
    show_image(ax, confocal_linescan, rasterized=True)

    # Add genotype labels
    add_label(ax, x_pos=0.02, y_pos=0.05, label='th;GFP', **style.txtLabelGPF)
    add_label(ax, x_pos=0.02, y_pos=0.15, label='atoh7;RFP', **style.txtLabelRFP)

    # Add neuropil label
    add_label(ax, x_pos=0.15, y_pos=0.72, label='superficial', color='white', size=6)
    add_label(ax, x_pos=0.7, y_pos=0.45, label='deep', color='white', size=6)

    # Add A-P axis
    add_ap_axis(ax, start=(0.05, 0.55), end=(0.05, 0.85), color='white', labels=("P", "A"))

    # ------------------------------------------------------------------------------------------------------------------
    # F) Line Scan Plot
    ax = axs_img['line_plot']
    plot_stratification_profile(line_plot_data, ax)
    y_pos = 1.02
    ax.text(7.5, y_pos, 'SAC', rotation=0, ha='center', va='center', **style.txtAnnotation)
    ax.text(22.5, y_pos, 'SGC', rotation=0, ha='center', va='center', **style.txtAnnotation)
    ax.text(60, y_pos, 'SFGS (1-6)', rotation=0, ha='center', va='center', **style.txtAnnotation)
    ax.text(95, y_pos, 'SM/SO', rotation=0, ha='center', va='center', **style.txtAnnotation)

    ax.set_xlabel("Neuropil position (%)")
    ax.set_ylabel("Norm. Intensity")
    ax.text(6.0, -0.25, 'deep', rotation=0, ha='center', va='center', **style.txtAnnotation)
    ax.text(95, -0.25, 'superficial', rotation=0, ha='center', va='center', **style.txtAnnotation)

    ax.set_xlim(0, 100.5)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # ------------------------------------------------------------------------------------------------------------------
    # G) HCR Images panels
    show_image(axs_img['hcr'], hcr, rasterized=True)
    add_ap_axis(axs_img['hcr'], start=(0.98, 0.1), end=(0.98, 0.4), color='white', labels=("P", "A"))

    # ------------------------------------------------------------------------------------------------------------------
    # H) HCR Percentage box plot
    # Define group order and labels
    ax = axs_img['hcr_per']
    groups = ["th+", "dat+", "vmat2+", "gad1+", "vglut2+", "drd1+", "drd2+"]

    group_positions = {g: i + 1 for i, g in enumerate(groups)}
    box_data = [hcr_data.loc[hcr_data["plot_label"] == g, "pct_positive_somas"].values for g in groups]
    bp = ax.boxplot(
        box_data,
        positions=[group_positions[g] for g in groups],
        widths=0.5,
        showfliers=False,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("black")

    # --- Overlay individual points, colored by larva ---
    all_larvae = sorted(hcr_data["larva_id"].unique())
    cmap = matplotlib.colormaps["tab20"].resampled(len(all_larvae))
    larva_colors = {larva: cmap(i) for i, larva in enumerate(all_larvae)}

    rng = np.random.default_rng(0)  # for reproducible jitter

    for g in groups:
        sub = hcr_data[hcr_data["plot_label"] == g]
        x_center = group_positions[g]
        for larva in sub["larva_id"].unique():
            larva_sub = sub[sub["larva_id"] == larva]
            jitter = rng.uniform(-0.20, 0.20, size=len(larva_sub))
            ax.scatter(
                x_center + jitter,
                larva_sub["pct_positive_somas"],
                color=larva_colors[larva],
                label=larva,
                edgecolor="black",
                linewidth=0.5,
                s=8,
                zorder=3,  # order of plotting
            )

    # --- Formatting ---
    ax.set_xticks([group_positions[g] for g in groups])
    ax.set_xticklabels(groups, rotation=45)
    ax.set_ylabel("% positive neurons")
    ax.set_xlabel("")
    hide_axis_spines(ax)

    # ==================================================================================================================
    # Add Sub-Figure Caps
    # from plotting_utils import set_caption_labels
    # set_caption_labels(fig)
    # plt.show()
    # exit()

    cap_pos = {
        'A': (np.float64(0.15), np.float64(0.75)),
        'B': (np.float64(0.38), np.float64(0.75)),
        'C': (np.float64(0.62), np.float64(0.75)),
        'D': (np.float64(0.15), np.float64(0.60)),
        'E': (np.float64(0.30), np.float64(0.60)),
        'F': (np.float64(0.48), np.float64(0.60)),
        'G': (np.float64(0.15), np.float64(0.45)),
        'H': (np.float64(0.55), np.float64(0.45)),
    }

    for cap, (fx, fy) in cap_pos.items():
        x, y = fx, fy
        fig.text(x, y, cap, **style.cap_style)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Store figures
    fig_dir = f"figs"
    matplotlib.rcParams["savefig.bbox"] = None

    # Add figure name to pdf
    fig.text(style.figure_label_pos[0], style.figure_label_pos[1], 'Figure 1', **style.figure_label)

    plt.savefig(f'{fig_dir}/01_figure_1.pdf', dpi=600)

    plt.close(fig)
    print('\n=== STORED FIGURE 1 TO HDD ====\n')


def main():
    # Create the figure
    plot_figure()


if __name__ == '__main__':
    main()

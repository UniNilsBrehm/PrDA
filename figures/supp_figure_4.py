"""
Assemble and export Supplementary Figure 4 for the manuscript:
A dopaminergic visuomotor pathway to the zebrafish optic tectum
Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann
iScience, Cell Press 2026

Supplementary Figure 4: Number of triggered swim bouts per stimulus type

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_style import PlotStyle, set_plot_defaults


def swim_response_count_histograms(
    axes,
    df,
    stim_labels=None,
):
    """
    Plot one discrete count histogram per stimulus.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Nine pre-existing axes arranged row-wise in a 3 × 3 grid.

    df : pandas.DataFrame
        Rows represent sweeps.
        Columns represent stimulus types.
        Values represent the number of swim bouts per sweep.

    stim_labels : sequence of str, optional
        Labels displayed above the stimulus panels.

    Returns
    -------
    numpy.ndarray
        Flattened array of plotted axes.
    """
    if stim_labels is None:
        stim_labels = [
            "DL", "GF", "GB",
            "GA", "GD", "BL",
            "DF", "MT", "BF",
        ]

    axes = np.asarray(axes, dtype=object).ravel()
    plot_df = df.copy()

    if len(axes) != len(stim_labels):
        raise ValueError(
            f"Expected {len(stim_labels)} axes, "
            f"but received {len(axes)}."
        )

    if plot_df.shape[1] != len(stim_labels):
        raise ValueError(
            f"Expected {len(stim_labels)} stimulus columns, "
            f"but received {plot_df.shape[1]}."
        )

    plot_df.columns = stim_labels

    all_values = plot_df.to_numpy(dtype=float)
    valid_values = all_values[~np.isnan(all_values)]

    if valid_values.size == 0:
        raise ValueError("The input DataFrame contains no valid values.")

    if np.any(valid_values < 0):
        raise ValueError("Swim-bout counts cannot be negative.")

    if not np.allclose(valid_values, np.rint(valid_values)):
        raise ValueError("Swim-bout values must be integer counts.")

    max_bouts = int(valid_values.max())
    bout_values = np.arange(max_bouts + 1)

    frequencies = {}

    for stim_label in stim_labels:
        values = (
            plot_df[stim_label]
            .dropna()
            .to_numpy(dtype=float)
        )

        values = np.rint(values).astype(int)

        frequencies[stim_label] = np.bincount(
            values,
            minlength=max_bouts + 1,
        )

    max_frequency = max(
        counts.max()
        for counts in frequencies.values()
    )

    y_max = max_frequency * 1.05 if max_frequency > 0 else 1

    for stim_idx, (ax, stim_label) in enumerate(
        zip(axes, stim_labels)
    ):
        row = stim_idx // 3
        col = stim_idx % 3

        ax.bar(
            bout_values,
            frequencies[stim_label],
            width=1.0,
            align="center",
            color="0.7",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.text(
            5, 15,
            stim_label,
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.5",
                linewidth=1,
                alpha=1
            ),
        )
        # Shared limits across all panels.
        ax.set_xlim(-1, max_bouts + 1)
        ax.set_ylim(0, y_max)

        ax.set_xticks(bout_values)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )

        # Only the left column shows y-axis labels.
        if col == 0:
            ax.set_ylabel("Number of sweeps")

        # Only the bottom row shows x-axis labels.
        if row == 2:
            ax.set_xlabel("Number of bouts")

        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_yticks([0, 5, 10, 15])
        ax.set_ylim(0, 17)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return axes


def plot_figure():
    set_plot_defaults()
    style = PlotStyle()

    base_dir = f'data/supp_fig4'

    response_heatmap_data = pd.read_csv(
        f'{base_dir}/swim_bouts_per_stimulus.csv',
        index_col=0
    )

    # Figure
    # A4 page: The figure will be on a DinA4 page in the final pdf file
    a4_w_cm, a4_h_cm = 21.0, 29.7
    fig_width = a4_w_cm / 2.54
    fig_height = a4_h_cm / 2.54

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Center the 16x20 cm content block on the A4 page
    content_w_cm, content_h_cm = 16, 16
    left = ((a4_w_cm - content_w_cm) / 2) / a4_w_cm
    right = 1 - left
    bottom = ((a4_h_cm - content_h_cm) / 2) / a4_h_cm
    top = 1 - bottom

    # reproduce the original subplots_adjust(0.05..0.95) margin, but inside the content box
    gs_left = left + 0.1 * (right - left)
    gs_right = left + 0.95 * (right - left)
    gs_bottom = bottom + 0.05 * (top - bottom)
    gs_top = bottom + 0.95 * (top - bottom)

    grid = matplotlib.gridspec.GridSpec(
        nrows=int(content_h_cm * 10), ncols=int(content_w_cm * 10),
        left=gs_left, right=gs_right, bottom=gs_bottom, top=gs_top,
        wspace=0, hspace=0,
    )

    axs = {}
    # ------------------------------------------------------------------
    # Panel A: 3 × 3 stimulus histograms
    # ------------------------------------------------------------------

    panel_a_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(
        nrows=3,
        ncols=3,
        subplot_spec=grid[0:160, 0:160],
        hspace=0.30,
        wspace=0.30,
    )

    # stim_labels = [
    #     "DL", "GF", "GB",
    #     "GA", "GD", "BL",
    #     "DF", "MT", "BF",
    # ]
    stim_labels = [
        "Dark Loom", "Grating Forward", "Grating Backward",
        "Grating Appears", "Grating Disappears", "Bright Loom",
        "Dark Flash", "Moving Target", "Bright Flash",
    ]

    axs["response_histograms"] = []

    for stim_idx, stim_label in enumerate(stim_labels):
        row = stim_idx // 3
        col = stim_idx % 3

        ax = fig.add_subplot(
            panel_a_grid[row, col],
            sharex=(
                axs["response_histograms"][0]
                if axs["response_histograms"]
                else None
            ),
            sharey=(
                axs["response_histograms"][0]
                if axs["response_histograms"]
                else None
            ),
        )

        axs["response_histograms"].append(ax)

    # Plot data into the already-created axes
    swim_response_count_histograms(
        axs["response_histograms"],
        response_heatmap_data,
        stim_labels=stim_labels
    )

    # ==================================================================================================================
    # Store Figure to HDD
    matplotlib.rcParams["savefig.bbox"] = None
    fig_dir = f"figs"
    fig_nr = 4

    fig.text(style.figure_label_pos[0], style.figure_label_pos[1], f'Suppl. Figure {fig_nr}', **style.figure_label)
    plt.savefig(f'{fig_dir}/11_supp_figure_{fig_nr}.pdf', dpi=600)
    plt.close(fig)
    print(f'\n=== STORED SUPP FIG {fig_nr} TO HDD ====\n')


def main():
    plot_figure()


if __name__ == "__main__":
    main()
    
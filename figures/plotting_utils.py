import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib import cm, colors, colorbar
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def add_label(ax, x_pos, y_pos, label, color, size, ha='left', va='center'):
    ax.text(
        x_pos, y_pos, label, transform=ax.transAxes, ha=ha, va=va, color=color, fontsize=size
    )


def add_ap_axis(
    ax,
    start=(0.2, 0.05),
    end=(0.8, 0.05),
    labels=("A", "P"),
    lw=1.5,
    fontsize=8,
    color="black"
):
    """
    Add a double-arrow anterior–posterior axis to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    start : tuple
        Start position in axes fraction coordinates.
    end : tuple
        End position in axes fraction coordinates.
    labels : tuple
        Labels at the ends of the arrow (default: ("A","P")).
    lw : float
        Line width of the arrow.
    fontsize : int
        Font size of labels.
    color : str
        Arrow and label color.
    """

    # draw double arrow
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="<->",
            lw=lw,
            color=color,
            mutation_scale=10
        )
    )

    # label positions
    ax.text(
        start[0],
        start[1]-0.02,
        labels[0],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color=color
    )

    ax.text(
        end[0],
        end[1]+0.01,
        labels[1],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color=color
    )


def set_caption_labels(fig):
    # Set Cap Label manually
    # Panel labels you want to assign in order
    labels = list("ABCDEFGHIJKLMNOPQRST")
    cap_pos = {}

    def onclick(event):
        if not labels:  # stop after last label
            print("All labels assigned.")
            return

        # Convert pixel coords → figure fraction
        x_fig, y_fig = fig.transFigure.inverted().transform((event.x, event.y))
        label = labels.pop(0)  # take next label
        cap_pos[label] = (round(x_fig, 2), round(y_fig, 2))

        # Draw the label on the figure
        fig.text(x_fig, y_fig, label, fontsize=14, fontweight="bold",
                 ha="center", va="center")
        fig.canvas.draw_idle()

        # Print current dictionary in copy-pasteable format
        print("cap_pos = {")
        for k, v in cap_pos.items():
            print(f"    '{k}': {v},")
        print("}")

    fig.canvas.mpl_connect('button_press_event', onclick)


def add_pictogram(pic, loc, ax, size=0.03, coord='data', ha='center', va='bottom'):
    """
    Parameters
    ----------
    pic:            the image
    loc:            position of the image
    ax:             the axes to plot
    size:           the size of the image
    coord:          data values: 'data', fraction: 'axes fraction'
    ha:             horizontal alignment
    va:             vertical alignment

    Returns
    -------

    """
    if ha == 'center':
        ha_val = 0.5
    elif ha == 'left':
        ha_val = 0
    else:
        ha_val = 1

    if va == 'bottom':
        va_val = 0
    elif ha == 'center':
        va_val = 0.5
    else:
        va_val = 1

    # 2. wrap it in OffsetImage
    img_artist = OffsetImage(pic, zoom=size, interpolation='none')  # zoom sets size

    # 3. choose where to anchor it – in *axes‐fraction* coords here
    #    (0,0)=bottom-left, (1,1)=top-right inside that axis
    # loc = (5*60.5, 1.05)

    ab = AnnotationBbox(img_artist, loc,
                        xycoords=coord,  # interpret loc in data values
                        # xycoords='axes fraction',  # interpret loc in fraction coords
                        box_alignment=(ha_val, va_val),  # (center, bottom)
                        frameon=False,  # no box around the image
                        pad=0)

    # 4. add it to whichever axis you want it floating over
    ax.add_artist(ab)


def add_row_label(ax, x_pos, y_pos, text, label_style):
    ax.text(x_pos, y_pos, text, transform=ax.transAxes, ha='left', va='center', **label_style)


def create_alternating_pattern(group_numbers):
    alternating = []
    current_value = 0  # Start with 0
    alternating.append(current_value)

    for i in range(1, len(group_numbers)):
        if group_numbers[i] != group_numbers[i - 1]:
            current_value = 1 - current_value  # Flip between 0 and 1
        alternating.append(current_value)

    return alternating


def find_integer_panel_positions(
    total_width_mm=160,
    num_panels=5,
    min_gap_mm=1,
    max_gap_mm=10
):
    """
    Finds integer panel widths and gaps fitting the total figure width,
    and returns positions in millimeters for each panel.
    """
    num_gaps = num_panels - 1
    solutions = []

    for s in range(min_gap_mm, max_gap_mm + 1):
        remaining_width = total_width_mm - num_gaps * s
        if remaining_width <= 0:
            continue
        if remaining_width % num_panels == 0:
            w = remaining_width // num_panels
            # compute positions:
            positions = []
            current_left = 0
            for i in range(num_panels):
                start = current_left
                end = start + w
                positions.append((start, end))
                current_left = end + s  # move to next
            solutions.append({
                "panel_width": w,
                "gap": s,
                "positions": positions
            })

    if solutions:
        for sol in solutions:
            print(f"\npanel width = {sol['panel_width']} mm, gap = {sol['gap']} mm")
            print("positions:")
            for idx, (start, end) in enumerate(sol['positions']):
                print(f"  Panel {idx+1}: [{start} : {end}] mm")
    else:
        print("No integer solutions found in this range.")


def add_colorbar(ax, label, label_font_size, label_rotation, label_pad, cmap, v_min, v_max, ticks=None):
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
    cb.ax.tick_params(labelsize=label_font_size)
    cb.set_label(label, rotation=label_rotation, labelpad=label_pad)
    if ticks is not None:
        cb.set_ticks(ticks)


def remove_y_axis(ax, ticks=True, label=True, complete=False):
    if complete:
        ax.yaxis.set_visible(False)  # Hide entire y-axis
    else:
        if ticks:
            ax.set_yticks([])  # Remove ticks
        if label:
            ax.set_ylabel('')  # Remove label


def remove_x_axis(ax, ticks=True, label=True, complete=False):
    if complete:
        ax.xaxis.set_visible(False)  # Hide entire y-axis
    else:
        if ticks:
            ax.set_xticks([])  # Remove ticks
        if label:
            ax.set_xlabel('')  # Remove label


def hide_axis_spines(ax, left=True, right=False, top=False, bottom=True):
    ax.spines['left'].set_visible(left)  # Hide left spine (y-axis line)
    ax.spines['right'].set_visible(right)  # Hide right spine
    ax.spines['top'].set_visible(top)  # Hide top spine (y-axis line)
    ax.spines['bottom'].set_visible(bottom)  # Hide bottom spine


def draw_sizebar(ax, size, label, loc, width, orientation='horizontal', font_size=10, color='black'):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from matplotlib.font_manager import FontProperties

    if orientation == 'horizontal':
        h_size = size
        v_size = width
    elif orientation == 'vertical':
        h_size = width
        v_size = size
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'.")

    asb = AnchoredSizeBar(ax.transData,
                          size=h_size,  # size of scale bar in x-axis units
                          label=label,
                          loc='lower left',  # the coordinate origin
                          pad=0.05,
                          borderpad=0.5,
                          sep=3,
                          color=color,
                          size_vertical=v_size,
                          frameon=False,
                          bbox_to_anchor=(loc[0], loc[1]),  # x, y
                          bbox_transform=ax.figure.transFigure,
                          # bbox_transform=ax.transData,
                          fontproperties=FontProperties(size=font_size)
                          )
    ax.add_artist(asb)


def plot_scale_bar(ax,bar_start_x, bar_length, bar_y, bar_label, lw, font_size, color='k'):
    # bar_start_x = 1  # x coordinate where bar starts
    # bar_y = 0  # y coordinate for bar
    # bar_length = 2  # duration in seconds
    # bar_label = '2 s'

    # Draw the scale bar
    ax.plot([bar_start_x, bar_start_x + bar_length], [bar_y, bar_y], color, lw=lw)  # horizontal bar

    # Add label
    ax.text(bar_start_x + bar_length / 2, bar_y - 0.1, bar_label, ha='center', va='top', fontsize=font_size)


def add_scale_bar(ax, size, label, location, orientation='horizontal', color='black', linewidth=2, fontsize=10,
                  padding=0.02):
    """
    Add a scale bar to a Matplotlib axis with support for relative coordinates.

    Parameters:
    - ax: Matplotlib axis where the scale bar will be added.
    - size: Length of the scale bar in data coordinates.
    - label: Label text for the scale bar.
    - location: Tuple (x, y) specifying the relative starting point of the scale bar (0 to 1 range).
    - orientation: 'horizontal' or 'vertical' for the scale bar's orientation.
    - color: Color of the scale bar.
    - linewidth: Thickness of the scale bar line.
    - fontsize: Font size of the label text.
    - padding: Padding (in relative coordinates) between the scale bar and the label.
    """
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Convert relative coordinates to data coordinates
    x_data = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y_data = ylim[0] + location[1] * (ylim[1] - ylim[0])

    # Convert padding from relative to data units
    padding_x = padding * (xlim[1] - xlim[0])
    padding_y = padding * (ylim[1] - ylim[0])

    if orientation == 'horizontal':
        # Draw the horizontal scale bar
        ax.plot([x_data, x_data + size], [y_data, y_data], color=color, linewidth=linewidth)
        # Add the label centered below the bar
        ax.text(x_data + size / 2, y_data - padding_y, label, ha='center', va='top', fontsize=fontsize, color=color)
    elif orientation == 'vertical':
        # Draw the vertical scale bar
        ax.plot([x_data, x_data], [y_data, y_data + size], color=color, linewidth=linewidth)
        # Add the label to the right of the bar
        ax.text(x_data + padding_x, y_data + size / 2, label, ha='left', va='center', fontsize=fontsize, color=color)
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'.")


def add_scale_bar_v2(ax, size, label, location, orientation='horizontal', color='black', linewidth=2, fontsize=10,
                  padding=0.02):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_data = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y_data = ylim[0] + location[1] * (ylim[1] - ylim[0])

    padding_x = padding * (xlim[1] - xlim[0])
    padding_y = padding * (ylim[1] - ylim[0])

    if orientation == 'horizontal':
        ax.plot([x_data, x_data + size], [y_data, y_data],
                color=color, linewidth=linewidth, clip_on=False)
        ax.text(x_data + size / 2, y_data - padding_y, label,
                ha='center', va='top', fontsize=fontsize, color=color, clip_on=False)
    elif orientation == 'vertical':
        ax.plot([x_data, x_data], [y_data, y_data + size],
                color=color, linewidth=linewidth, clip_on=False)
        ax.text(x_data + padding_x, y_data + size / 2, label,
                ha='left', va='center', fontsize=fontsize, color=color, clip_on=False)

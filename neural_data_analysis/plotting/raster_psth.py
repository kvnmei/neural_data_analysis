from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Range1d


def plot_raster_psth(raster_df, psth_df, plot_params, backend="bokeh"):
    """
    Plots a raster and PSTH for spikes to an event. All raster/PSTH are for a single neuron's firing rate.

    Args:
        raster_df (pd.DataFrame): Each row is an event.
            Expected columns:
                event_id (int): The event ID.
                spike_times (np.ndarray): The relative spike time of the neuron within the window of interest around
                 the event time.
                category (str): The category of the event.
                color (str): Color to plot for the category.
        psth_df (pd.DataFrame): Each row is a category.
            Expected columns:
                bin_centers (float): The center of the bin in the PSTH.
                firing_rate (float): The firing rate of the neuron in the bin.
                category (str): The name of the category.
                color (str): Color to plot for the category.
        plot_params (dict): A dictionary of parameters for plotting.
            Expected keys:
                raster_title (str): The title of the raster plot.
                psth_title (str): The title of the PSTH plot.
                xlabel (str): The x-axis label.
                ylabel (str): The y-axis label.
                save_path (str): The path to save the plot to.
                figsize (tuple): The size of the figure.
                colors (list): The colors to use for the raster plot.

        backend (str): What plotting package to use for plotting. Options are "bokeh"...

    Returns:

    """
    if backend == "bokeh":
        # Flatten the data for Bokeh plotting
        flattened_spike_times = []
        flattened_event_ids = []
        flattened_colors = []
        flattened_categories = []
        for _, raster_row in raster_df.iterrows():
            for spike_time in raster_row['spike_times']:
                flattened_spike_times.append(spike_time)
                flattened_event_ids.append(raster_row['new_event_id'])
                flattened_colors.append(raster_row['color'])
                flattened_categories.append(raster_row['category'])

        # Create a ColumnDataSource with flattened data
        source = ColumnDataSource({
            'spike_times': flattened_spike_times,
            'event_id': flattened_event_ids,
            'color': flattened_colors,
            'category': flattened_categories
        })

        min_event_id = min(flattened_event_ids)
        max_event_id = max(flattened_event_ids)

        # Create figure for raster plot
        raster_plot = figure(title=plot_params.get("raster_title", "Raster Plot"),
                             x_axis_label=plot_params.get("xlabel", "Time (s)"),
                             y_axis_label=plot_params.get("ylabel", "Event"),
                            width=plot_params.get('plot_width', 800),
                            height=plot_params.get('plot_height', 400),
                             y_range=Range1d(start=max_event_id + 1, end=min_event_id - 1),
                             )

        # Plot using the new source
        raster_plot.dash('spike_times', 'event_id', source=source, color='color', size=15, angle=90,
                         angle_units="deg", legend_field="category", line_width=2)

        # Adjust the legend
        raster_plot.legend.title = 'Category'
        raster_plot.legend.orientation = "vertical"
        raster_plot.legend.location = "top_right"
        raster_plot.add_layout(raster_plot.legend[0], 'right')
        psth_plot = row(raster_plot)

        # Create figure for PSTH plot
        psth_plot = figure(title=plot_params.get("psth_title", "PSTH"),
                           x_axis_label=plot_params.get("xlabel", "Time (s)"),
                           y_axis_label="Firing rate (spikes/s)",
                           width=plot_params.get('plot_width', 800),
                           height=plot_params.get('plot_height', 400),
                           )

        rows = []
        for _, psth_row in psth_df.iterrows():
            for bin_center, firing_rate in zip(psth_row['bin_centers'], psth_row['firing_rate']):
                rows.append({
                    'bin_center': bin_center,
                    'firing_rate': firing_rate,
                    'category': psth_row['category'],
                    'color': psth_row['color']
                })

        expanded_psth_df = pd.DataFrame(rows)
        for category, group in expanded_psth_df.groupby('category'):
            source = ColumnDataSource(group)
            psth_plot.line('bin_center', 'firing_rate', source=source, line_width=2, color=group['color'].iloc[0], legend_label=str(category))

        psth_plot.legend.title = 'Category'
        psth_plot.legend.orientation = "vertical"
        psth_plot.legend.location = "top_right"

        psth_plot.add_layout(psth_plot.legend[0], 'right')
        psth_plot = row(psth_plot)

        # Combine plots vertically
        combined = column(raster_plot, psth_plot)

        # Output to file or notebook
        output_file(plot_params.get('save_path', "raster_psth_plot.html"))

        show(combined)



def plot_raster_psth2(spike_times, psth, bin_centers, time_window, bin_size, step_size, plot_params, backend: str= "seaborn"):
    """
    Plots a raster and PSTH for spikes to an event. All raster/PSTH are for a single neuron's firing rate.

    Args:
        spike_times (np.ndarray): (n_events, n_spikes)
            For every event, the relative spike times of the neuron within the window of interest around the event time.
        psth (np.ndarray): (n_bins,)
            The PSTH for the neuron. For every bin, the firing rate of the neuron.
        bin_centers (np.ndarray): The center of each bin in the PSTH.
        time_window (list): The time window around the event time to plot the raster/PSTH.
        bin_size (float): The size of each bin in the PSTH.
        step_size (float): The step size between each bin in the PSTH.
        plot_params (dict): A dictionary of parameters for plotting.
            Expected keys:
                title (str): The title of the plot.
                subtitle (str): The subtitle of the plot.
                xlabel (str): The x-axis label.
                ylabel (str): The y-axis label.
                save_path (str): The path to save the plot to.
                figsize (tuple): The size of the figure.
                colors (list): The colors to use for the raster plot.
        backend (str): What plotting package to use for plotting. Options are "seaborn" and "bokeh".

    Returns:
        None

    Example:
        spike_times =
        plot_raster_psth(
            spike_times=spike_times,
            psth=psth,
            time_window=[-1, 1],
            bin_size=0.1,
            step_size=0.05,
            plot_params=plot_params,
            backend="seaborn",
        )
    """

    # this code is to handle the case where there is only one event
    if not isinstance(spike_times[0], np.ndarray):
        spike_times = np.array([spike_times])

    total_events = len(spike_times)
    assert len(bin_centers) == len(psth), "Length of bins and PSTH must be the same."

    if backend == "seaborn":
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")

        fig, axes = plt.subplots(
            figsize=plot_params.get("figsize", (12, 8)),
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.suptitle(
            f"{plot_params.get('title', 'Raster Plot with PSTH')}\n"
            f"{plot_params.get('subtitle', None)}\nBin size {bin_size} s, Step size {step_size} s"
        )
        plt.subplots_adjust(top=0.85)

        axes[0].eventplot(spike_times, linewidths=0.5)
        axes[0].set_yticks(np.arange(total_events))
        axes[0].set_yticklabels(np.arange(1, total_events + 1))
        # ax[0].set_yticklabels(np.arange(1, total_events + 1)[::-1]) # for reverse order
        axes[0].set_xlabel(plot_params.get("xlabel", "Time (s)"))
        axes[0].set_ylabel(plot_params.get("ylabel", "Event"))
        # ax[0].set_title(plot_params.get('title', "Raster Plot with PSTH"))

        axes[1].plot(
            bin_centers,
            psth,
            linewidth=1,
        )
        axes[1].set_xlim([time_window[0], time_window[1]])
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Firing rate (spikes/s)")

        # plot the stimulus time as a vertical line
        axes[0].axvline(x=0, color="black", linestyle="--")
        axes[1].axvline(x=0, color="black", linestyle="--")

        # for time in plot_params["scene_cut_times"]:
        #     ax[1].axvline(x=time, color="green", linestyle="-", linewidth=0.5)
        #
        # for time in plot_params["scene_change_times"]:
        #     ax[1].axvline(x=time, color="orange", linestyle="-", linewidth=0.75)

        plt.savefig(f"{plot_params['save_path']}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    elif backend == "bokeh":
        # Prepare color map for event groups
        unique_groups = np.unique(event_groups)
        colors = plot_params.get('colors', ['blue', 'green', 'red', 'purple', 'orange', 'yellow'])
        color_map = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}

        # Create Bokeh figure for raster plot
        p = figure(title=plot_params.get("title", "Raster Plot with PSTH"),
                   x_axis_label=plot_params.get("xlabel", "Time (s)"),
                   y_axis_label=plot_params.get("ylabel", "Event"))

        # Plot each event in the raster plot
        for i, (event, group) in enumerate(zip(spike_times, event_groups)):
            group_color = color_map[group]
            p.circle(event, np.full(event.shape, i), size=5, color=group_color, legend_label=str(group))

        # Plot PSTH (You might need to adjust this part according to your PSTH data structure)
        p.line(np.arange(time_window[0], time_window[1], step_size), psth, line_width=2)

        p.legend.title = 'Event Groups'
        p.legend.location = "top_right"

        # Output to static HTML file (or use output_notebook() for inline plotting)
        output_file(plot_params.get('save_path', "raster_psth_plot.html"))

        show(p)


def compute_psth(
    spike_times: np.ndarray,
    time_range: List[float],
    bin_size: float = 1.0,
    step_size: float = 0.5,
) -> np.ndarray:
    """
    Compute the peri-stimulus time histogram values over the time range,
    by using a sliding window of size bin_size and step size step_size.
    The spike times and time range should be relative to the same type of event (e.g., image onset).

    Args:
        spike_times (numpy array): shape (n_neurons, n_spike_times)
            array of single neuron spike times, relative to event
        time_range (list[float]): [start, end]
            start and end time for PSTH, relative to event
        bin_size (float): size of the window to calculate the firing rate over
        step_size (float): step size with which to move the sliding window

    Returns:
        psth (numpy array): shape (n_neurons, n_bins)
            values of firing rates over the time range

    Example:
        Calculate the firing rate array from -0.5 to 1.5 seconds from when a stimulus image appears,
        with a bin size of 0.500 seconds and step size of 0.250 seconds.
        psth = compute_psth(spike_times, time_range=[-0.5, 1.5], bin_size=0.500, step_size=0.250)
    """
    if not isinstance(spike_times[0], np.ndarray):
        spike_times = np.array([spike_times])

    total_events = len(spike_times)
    num_bins = int(np.ceil((time_range[1] - time_range[0] - bin_size) / step_size)) + 1
    psth = np.zeros(num_bins)

    for j in np.arange(num_bins):
        bin_start = (j * step_size) + time_range[0]
        bin_end = bin_start + bin_size
        for i in np.arange(total_events):
            spike_mask = np.logical_and(
                spike_times[i] >= bin_start, spike_times[i] < bin_end
            )
            spike_count = np.sum(spike_mask)
            psth[j] += spike_count
    psth /= total_events * bin_size
    return psth

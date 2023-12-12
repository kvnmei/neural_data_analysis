import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pandas as pd

import bokeh.layouts
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Range1d, HoverTool


def compute_psth(
    spike_times: np.ndarray,
    time_range: list[float],
    bin_size: float = 1.0,
    step_size: float = 0.5,
) -> np.ndarray:
    """
    Compute the peri-stimulus time histogram values over the time range,
    by using a sliding window of size bin_size and step size step_size.
    The spike times and time range should be relative to the same type of event (e.g., image onset).

    Args:
        spike_times (np.ndarray): shape (n_neurons, n_spike_times)
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


def compute_psth_per_category(
    raster_df, time_range: list[float], bin_size: float = 1, step_size: float = 0.5
) -> dict:
    """
    Compute the PSTH values for each category over the time range,
    using a sliding window of size bin_size and step size step_size.

    Args:
        raster_df (pd.DataFrame): DataFrame with columns 'event_id', 'spike_times', 'category', 'color'
        time_range (list[float]): [start, end] start and end time for PSTH, relative to event
        bin_size (float): size of the window to calculate the firing rate over
        step_size (float): step size with which to move the sliding window

    Returns:
        dict: A dictionary with categories as keys and PSTH arrays as values
    """

    # Get unique categories
    unique_categories = raster_df["category"].unique()

    # Initialize a dictionary to hold PSTH data for each category
    psth_per_category = {}

    # Iterate over each category
    for category in unique_categories:
        # Extract spike times for the current category
        category_spike_times = np.array(
            raster_df[raster_df["category"] == category]["spike_times"].to_list()
        )

        # Compute PSTH for the current category
        psth = compute_psth(category_spike_times, time_range, bin_size, step_size)

        # Store the result in the dictionary
        psth_per_category[category] = psth

    return psth_per_category


def plot_raster_psth(
    raster_df: pd.DataFrame,
    psth_df: pd.DataFrame,
    plot_params: dict,
    backend: str = "bokeh",
    output_plot=False,
) -> None:
    """
    Plots a raster and PSTH for spikes to an event. All raster/PSTH are for a single neuron's firing rate.

    NOTE: This function can be used to plot population activity as well. Instead of every row being a single neuron's
    activity for a single even, each row would be a neuron's activity for the same event.
    The PSTH would be the average firing rate across neurons for each bin, and may be grouped by category that neurons
     belong to (e.g., by brain region).

    Args:
        raster_df (pd.DataFrame): Each row is an event or trial.
            Expected columns:
                event_id (int): The event ID.
                spike_times (np.ndarray): (n_spikes,)
                    The relative spike time of the neuron within the window of interest around
                    the event time.
                category (str): The category of the event/trial.
                color (str): Color to plot for the category.
        psth_df (pd.DataFrame): Each row is a PSTH for a category of trials.
            Expected columns:
                bin_centers (list[float]): The center of the bin in the PSTH.
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

        backend (str): What plotting package to use for plotting. Options are "bokeh"...

    Returns:

    """
    if backend == "bokeh":
        # Flatten the data for Bokeh plotting from wide format to long format
        raster_data_long = []
        for _, raster_row in raster_df.iterrows():
            for spike_time in raster_row["spike_times"]:
                raster_data_long.append(
                    {
                        "spike_times": spike_time,
                        "event_id": raster_row["event_id"],
                        "color": raster_row["color"],
                        "category": raster_row["category"],
                    }
                )
        raster_df_long = pd.DataFrame(raster_data_long)
        source = ColumnDataSource(raster_df_long)

        # Create figure for raster plot
        min_event_id = min(raster_df_long["event_id"])
        max_event_id = max(raster_df_long["event_id"])
        raster_plot = figure(
            title=plot_params.get("raster_title", "Raster Plot"),
            x_axis_label=plot_params.get("xlabel", "Time (s)"),
            y_axis_label=plot_params.get("ylabel", "Event"),
            width=plot_params.get("plot_width", 800),
            height=plot_params.get("plot_height", 1600),
            y_range=Range1d(start=max_event_id + 1, end=min_event_id - 1),
        )

        # Plot raster
        raster_plot.dash(
            x="spike_times",
            y="event_id",
            source=source,
            color="color",
            size=15,
            angle=90,
            angle_units="deg",
            legend_field="category",
            line_width=2,
        )

        # Adjust the legend
        raster_plot.legend.title = "Category"
        raster_plot.legend.orientation = "vertical"
        raster_plot.legend.location = "top_right"
        raster_plot.add_layout(raster_plot.legend[0], "right")
        raster_plot = bokeh.layouts.row(raster_plot)

        # Create figure for PSTH plot
        psth_plot = figure(
            title=plot_params.get("psth_title", "PSTH"),
            x_axis_label=plot_params.get("xlabel", "Time (s)"),
            y_axis_label="Firing rate (spikes/s)",
            width=plot_params.get("plot_width", 800),
            height=plot_params.get("plot_height", 400),
        )

        # Flatten from a wide format (where each row contains the entire series of bin centers and firing rates for a
        # category) to a long format (where each row represents a single data point with a bin center, firing rate,
        # category, and color).
        psth_data_long = []
        for i, psth_row in psth_df.iterrows():
            for bin_center, firing_rate in zip(
                psth_row["bin_centers"], psth_row["firing_rate"]
            ):
                psth_data_long.append(
                    {
                        "bin_center": bin_center,
                        "firing_rate": firing_rate,
                        "category": psth_row["category"],
                        "color": psth_row["color"],
                    }
                )
        psth_df_long = pd.DataFrame(psth_data_long)

        # Plot PSTH
        for color, group in psth_df_long.groupby("color"):
            source = ColumnDataSource(group)
            psth_plot.line(
                x="bin_center",
                y="firing_rate",
                source=source,
                line_width=2,
                color=color,
                legend_label=str(group["category"].iloc[0]),
            )

        # Adjust the legend
        psth_plot.legend.title = "Category"
        psth_plot.legend.orientation = "vertical"
        psth_plot.legend.location = "top_right"
        psth_plot.add_layout(psth_plot.legend[0], "right")
        psth_plot = bokeh.layouts.row(psth_plot)

        if output_plot:
            # Combine plots vertically
            combined = column(raster_plot, psth_plot)
            # Output to file or notebook
            output_file(f"{plot_params.get('save_path', 'raster_psth_plot')}.html")
            show(combined)
        else:
            return raster_plot, psth_plot

    elif backend == "seaborn":
        sns.set(style="white", context="paper")

        # Creating subplots
        # noinspection PyTypeChecker
        fig, axes = plt.subplots(
            figsize=plot_params.get("figsize", (12, 8)),
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # Raster plot
        axes[0].eventplot(
            data=raster_df,
            positions="spike_times",
            lineoffsets=1,
            linelengths=1,
            linewidths=2,
            colors="color",
        )
        axes[0].set_ylabel(plot_params.get("ylabel", "Event"))
        axes[0].set_title(plot_params.get("raster_title", "Raster Plot"))
        axes[0].set_yticks(np.arange(len(raster_df)))
        # axes[0].set_yticklabels(np.arange(1, len(raster_df) + 1))
        axes[0].set_yticklabels(
            np.arange(1, len(raster_df) + 1)[::-1]
        )  # for reverse order

        unique_categories = (
            raster_df[["category", "color"]].drop_duplicates().sort_values("category")
        )
        # Creating custom legend for the raster plot
        legend_handles = [
            mpatches.Patch(color=row["color"], label=f"{row['category']}")
            for _, row in unique_categories.iterrows()
        ]

        axes[0].legend(
            handles=legend_handles,
            title="Category",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # PSTH plot
        for _, psth_row in psth_df.iterrows():
            axes[1].plot(
                psth_row["bin_centers"],
                psth_row["firing_rate"],
                label=str(psth_row["category"]),
                color=psth_row["color"],
            )
        axes[1].set_xlabel(plot_params.get("xlabel", "Time (s)"))
        axes[1].set_ylabel("Firing rate (spikes/s)")
        axes[1].set_title(plot_params.get("psth_title", "PSTH"))
        axes[1].legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.suptitle(
            f"{plot_params.get('title', 'Raster Plot with PSTH')}\n"
            f"{plot_params.get('subtitle', None)}\n"
        )
        # plot the stimulus time as a vertical line
        axes[0].axvline(x=0, color="black", linestyle="--")
        axes[1].axvline(x=0, color="black", linestyle="--")

        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(f"{plot_params.get('save_path', 'raster_psth_plot')}.png", dpi=300)
        plt.close(fig)

        # for time in plot_params["scene_cut_times"]:
        #     ax[1].axvline(x=time, color="green", linestyle="-", linewidth=0.5)
        #
        # for time in plot_params["scene_change_times"]:
        #     ax[1].axvline(x=time, color="orange", linestyle="-", linewidth=0.75)


def plot_gantt_bar_chart(
    gantt_df: pd.DataFrame,
    plot_params: dict,
    backend: str = "bokeh",
    output_plot: bool = False,
) -> None:
    """
    Plot a Gantt bar chart with Bokeh.

    Args:
        gantt_df (pd.DataFrame): Each row is a time point.
            Expected columns:
                time_index (int): The index of the time point.
                category (str): The category of the time point.
        plot_params (dict): A dictionary of parameters for plotting.
            Expected keys:
                title (str): The title of the plot.
                subtitle (str): The subtitle of the plot.
                xlabel (str): The x-axis label.
                ylabel (str): The y-axis label.
                save_path (str): The path to save the plot to.
                figsize (tuple): The size of the figure.
                colors (list): The colors to use for the raster plot.
        backend (str): The plotting package to use.

    Returns:

    """
    if backend == "bokeh":
        df = gantt_df.copy()
        # Determine where category changes occur
        df["change"] = df["category"].ne(df["category"].shift())
        df["start_time"] = df["time_index"]
        df["end_time"] = (
            df["time_index"].shift(-1).fillna(df["time_index"].iloc[-1] + 1)
        )

        # Filter rows where category changes
        blocks = df[df["change"]].copy()
        # end rows are the times right before the category changes, so shift the "change" column by -1 and
        # get those times
        end_rows = df[df["change"].shift(-1).fillna(True)].copy()
        blocks["end_time"] = end_rows["end_time"].values

        unique_categories = blocks["category"].unique()
        category_y_positions = {
            cat: i for i, cat in enumerate(sorted(unique_categories), start=0)
        }
        blocks["y_bottom"] = blocks["category"].map(category_y_positions) - 0.4
        blocks["y_top"] = (
            blocks["y_bottom"] + 0.8
        )  # Adjust the height of the bars as needed

        # Map categories to colors (adjust as needed)
        color_map = {0: "blue", 1: "green", 2: "red"}
        blocks["color"] = blocks["category"].map(color_map)

        # Creating a ColumnDataSource
        source = ColumnDataSource(blocks)

        # Create the figure
        p = figure(
            height=350,
            title=plot_params.get("title", "Intervals Bar Chart"),
            toolbar_location=None,
            tools="",
            y_range=(-1, len(unique_categories)),
            x_axis_label=plot_params.get("xlabel", "Time Index"),
            y_axis_label=plot_params.get("ylabel", "Category"),
        )

        # Add quad glyphs
        p.quad(
            source=source,
            left="start_time",
            right="end_time",
            bottom="y_bottom",
            top="y_top",
            fill_color="color",
            line_color="black",
        )

        # Customizing the plot
        # p.yaxis.ticker = list(
        #     map(lambda x: x + 0.4, list(category_y_positions.values()))
        # )
        p.yaxis.ticker = list(category_y_positions.values())
        p.yaxis.major_label_overrides = {
            v: f"Category {k}" for k, v in category_y_positions.items()
        }
        p.xgrid.grid_line_color = None

        # Adding hover tool
        hover = HoverTool(
            tooltips=[
                ("Category", "@category"),
                ("Start Frame", "@start_time"),
                ("End Frame", "@end_time"),
            ]
        )
        p.add_tools(hover)

        if output_plot:
            # Output to file
            output_file(f"{plot_params.get('save_path', 'gantt_plot')}.html")
            show(p)
        else:
            return p

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from pathlib import Path


def plot_neuron_firing_rate(
    neuron_fr_df: pd.DataFrame,
    smooth_firing_rate: bool = False,
    plot_timepoint_mask: bool = False,
    plot_timepoint_label: bool = False,
    show_boxplot: bool = False,
    show_violinplot: bool = False,
    sigma: int = 25,
    **kwargs,
):
    """
    Plot the firing rate of a neuron over time with an indicator for specific timepoints.

    Parameters:
        firing_rate (np.ndarray): Array of firing rates for a single neuron.
        smooth_firing_rate (bool): If True, smooth the firing rate using a Gaussian filter.
        add_timepoint_labels (bool): If True, highlight specific timepoints in the plot.
        word (str): The word label to highlight in the plot.
        cell_id (str): The ID of the neuron to plot.
        show_boxplot (bool): If True, displays a box plot of firing rates based on word presence.
        show_violinplot (bool): If True, displays a violin plot of firing rates based on word presence.
        sigma (int): The sigma value for Gaussian smoothing of the firing rate.

    Optional kwargs:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        xtick_labels (list, optional): Custom labels for the x-axis ticks.
        ytick_labels (list, optional): Custom labels for the y-axis ticks.
        save_dir (str or Path, optional): Directory to save the plot.
        save_filename (str, optional): File name to save the plot.


    Returns:
        None
    """
    # Set the default save directory if not provided
    save_dir = kwargs.get("save_dir", Path("plots"))
    save_dir = Path(save_dir)  # Ensure save_dir is a Path object
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set the default save file name if not provided
    save_filename = kwargs.get("save_filename", "neuron_firing_rate.png")
    xlabel = kwargs.get("xlabel", "Timepoint")
    ylabel = kwargs.get("ylabel", "Firing Rate (Hz)")
    title = kwargs.get("title", "Firing Rate (Hz)")

    # Extract data from neuron_fr_df
    timepoints = neuron_fr_df["timepoint"]
    firing_rate = neuron_fr_df["firing_rate"]
    timepoint_mask = neuron_fr_df["timepoint_mask"]
    timepoint_label = neuron_fr_df["timepoint_label"]
    cell_id = neuron_fr_df["cell_id"].values[0]
    if plot_timepoint_label:
        label_name = neuron_fr_df["timepoint_label_name"].values[0]

    sigma = 25
    if smooth_firing_rate:
        firing_rate_smoothed = np.zeros(len(firing_rate))
        firing_rate_smoothed = gaussian_filter1d(firing_rate, sigma=sigma)
        firing_rate = firing_rate_smoothed
        ylabel = kwargs.get("ylabel", f"Smoothed Firing Rate (Hz) (sigma = {sigma})")

    # --------------------------- OPTION 1: TWO SUBPLOTS ---------------------------
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )

    # Plot firing rate on ax1
    ax1.plot(
        timepoints,
        firing_rate,
        linestyle="-",
        color="blue",
        alpha=0.7,
        label=ylabel,
    )

    # Plot masked frames on ax1
    # if "timepoint_mask" in neuron_fr_df.columns:
    if plot_timepoint_mask:
        y_min, y_max = ax1.get_ylim()
        ax1.fill_between(
            timepoints,
            y_min,
            y_max,
            where=timepoint_mask,
            color="black",
            alpha=0.7,
            label="Masked Frames",
        )

    ax1.set_ylabel("Firing Rate (Hz)")
    ax1.legend(loc="upper right")
    ax1.tick_params(labelbottom=False)

    # Plot timepoint_label on ax2
    if plot_timepoint_label:
        ax2.plot(
            timepoints,
            timepoint_label,
            drawstyle="steps-post",
            color="red",
            label="Label Presence",
        )
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_ylabel(f"{label_name.title()} Label Presence")
        ax2.legend(loc="upper right")

        ax2.set_xlabel("Timepoint")

    # Set main title
    plt.suptitle(f"Neuron Firing Rate Over Time with '{label_name}' Label Presence\n{cell_id}")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_filename:
        # Create the plots directory if it doesn't exist
        fig.savefig(
            save_dir
            / (Path(save_filename).stem + "_subplots" + Path(save_filename).suffix)
        )

    # Show plot
    plt.show()

    # ------------------------- OPTION 2: OVERLAY PLOTS -------------------------
    # Plot the firing rate over time
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the firing rate
    ax.plot(
        timepoints,
        firing_rate,
        linestyle="-",
        color="blue",
        alpha=0.7,
        label="Firing Rate",
    )

    # Plot the masked frames
    if plot_timepoint_mask:
        y_min, y_max = ax.get_ylim()
        ax.fill_between(
            timepoints,
            y_min,
            y_max,
            where=timepoint_mask,
            color="black",
            alpha=0.7,
            label="Masked Frames",
        )

    # Indicate the timepoint_label
    if plot_timepoint_label:
        # Shade the area above the firing rate where the label is present
        ax.fill_between(
            timepoints,
            firing_rate,
            firing_rate.max(),
            where=timepoint_label.astype(bool),
            color="red",
            alpha=0.3,
            label=f"'{label_name}' Label Present",
        )

    # Customize plot
    ax.set_title(f"Neuron Firing Rate Over Time with '{label_name}' Label Presence\n{cell_id}")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_filename:
        # Create the plots directory if it doesn't exist
        fig.savefig(
            save_dir
            / (Path(save_filename).stem + "_overlay" + Path(save_filename).suffix)
        )
    plt.show()

    # ------------------------- OPTION 3: SECONDARY Y-AXIS -------------------------
    # Plot the firing rate over time
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the firing rate on the primary y-axis
    ax1.plot(
        timepoints,
        firing_rate,
        linestyle="-",
        color="blue",
        alpha=0.8,
        label="Firing Rate",
    )
    ax1.set_ylabel(ylabel, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create a secondary y-axis for timepoint_label
    if plot_timepoint_label:
        ax2 = ax1.twinx()

        # Plot the timepoint_label on the secondary y-axis
        ax2.plot(
            timepoints,
            timepoint_label,
            drawstyle="steps-post",
            color="red",
            alpha=0.5,
            label=f"'{label_name}' Label Presence",
        )
        ax2.set_ylabel(f"'{label_name}' Label Presence", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim(-0.1, 1.1)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Set the x-axis label and title
    ax1.set_xlabel("Timepoint")
    ax1.set_title(f"Neuron Firing Rate Over Time with '{label_name}' Label Presence\n{cell_id}")
    plt.tight_layout()
    if save_filename:
        # Create the plots directory if it doesn't exist
        fig.savefig(
            save_dir
            / (
                Path(save_filename).stem
                + "_secondary-yaxis"
                + Path(save_filename).suffix
            )
        )
    plt.show()
    #
    # # Plot the firing rate over time
    # fig, ax = plt.subplots(figsize=(12, 6))
    #
    # # Plot original firing rate
    # ax.plot(
    #     timepoints,
    #     firing_rate,
    #     linestyle="-",
    #     color="blue",
    #     alpha=0.3,
    #     label="Original Firing Rate",
    # )
    # if smooth_firing_rate:
    #     # Plot smoothed firing rate
    #     ax.plot(
    #         frames,
    #         cell_fr_smooth,
    #         linestyle="-",
    #         color="black",
    #         label="Smoothed Firing Rate",
    #     )
    # if timepoint_labels is not None:
    #     # Highlight the presence of the word
    #     ax.fill_between(
    #         timepoints,
    #         firing_rate.max(),
    #         where=timepoint_labels,
    #         color="red",
    #         alpha=1,
    #         label=f"Word '{word}' Present",
    #     )
    #
    # if timepoint_mask is not None:
    #     ax.fill_between(
    #         timepoints,
    #         ax.get_ylim()[1],
    #         where=timepoint_mask > 0,
    #         color="gray",
    #         alpha=0.5,
    #         label="Masked Frames",
    #     )
    #
    # # Customize plot
    # ax.set_title(f"Neuron {cell_id} in {brain_area}")
    # ax.set_xlabel("Frame")
    # ax.set_ylabel("Firing Rate (Hz)")
    # ax.legend(loc="upper right")
    # plt.tight_layout()
    # plt.show()
    #
    # # Optionally plot box plot and violin plot
    # if show_boxplot or show_violinplot:
    #     # Prepare data for plotting
    #     # Convert word presence to a categorical variable
    #     single_unit_df[f"{word}_present"] = single_unit_df[f"{word}_present"].fillna(0)
    #     single_unit_df[f"{word}_present"] = single_unit_df[f"{word}_present"].astype(
    #         int
    #     )
    #     single_unit_df = single_unit_df.dropna(subset=["firing_rate"])
    #     single_unit_df[f"{word}_present"] = single_unit_df[f"{word}_present"].map(
    #         {0: "Absent", 1: "Present"}
    #     )
    #
    #     if show_boxplot:
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #         sns.boxplot(
    #             x=f"{word}_present", y="firing_rate", data=single_unit_df, ax=ax
    #         )
    #         ax.set_title(f"Firing Rate by Word Presence for Neuron {cell_id}")
    #         ax.set_xlabel(f"Word '{word}' Presence")
    #         ax.set_ylabel("Firing Rate (Hz)")
    #         plt.tight_layout()
    #         plt.show()
    #
    #     if show_violinplot:
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #         sns.violinplot(
    #             x=f"{word}_present", y="firing_rate", data=single_unit_df, ax=ax
    #         )
    #         ax.set_title(
    #             f"Firing Rate Distribution by Word Presence for Neuron {cell_id}"
    #         )
    #         ax.set_xlabel(f"Word '{word}' Presence")
    #         ax.set_ylabel("Firing Rate (Hz)")
    #         plt.tight_layout()
    #         plt.show()

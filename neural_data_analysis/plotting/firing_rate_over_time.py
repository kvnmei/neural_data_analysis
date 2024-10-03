import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


def plot_neuron_firing_rate(
    firing_rate: np.ndarray,
    video_data,
    smooth_firing_rate: bool = False,
    add_label: bool = False,
    word: str = "gun",
    cell_id: str = None,
    show_boxplot: bool = False,
    show_violinplot: bool = False,
    sigma: int = 25,
    **kwargs,
):
    """
    Plot the firing rate of a neuron over time with an indicator for specific timepoints.

    Parameters:
        firing_rate (np.ndarray): Array of firing rates for a single neuron.
        video_data: Video data object containing embeddings and masking information.
        smooth_firing_rate (bool): If True, smooth the firing rate using a Gaussian filter.
        add_highlights (bool): If True, highlight specific timepoints in the plot.
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

    # Get the index of the word in the embeddings
    try:
        word_idx = video_data.blip2_words_df[
            video_data.blip2_words_df["word"] == word
        ].index[0]
    except IndexError:
        raise ValueError(
            f"The word '{word}' was not found in the video data embeddings."
        )

    if smooth_firing_rate:
        cell_fr_smooth = np.zeros(len(cell_fr))
        cell_fr_smooth[video_data.masked_frames_index] = gaussian_filter1d(
            cell_fr_masked, sigma=sigma
        )

    if add_highlights:
        # Create a binary vector indicating the presence of the word
        timepoints = np.arange(len(cell_fr))
        binary_mask = np.zeros(len(cell_fr))
        binary_mask[video_data.masked_frames_index] = video_data.embeddings["blip2"][
            :, word_idx
        ]
        binary_mask[~video_data.masked_frames_index] = np.nan

    # Create a DataFrame for plotting and analysis
    single_unit_df = pd.DataFrame(
        {
            "frame": frames,
            "firing_rate": cell_fr,
            "smoothed_firing_rate": cell_fr_smooth,
            f"{word}_present": binary_mask,
        }
    )

    # Plot the firing rate over time
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot original firing rate
    ax.plot(
        frames,
        cell_fr,
        linestyle="-",
        color="blue",
        alpha=0.3,
        label="Original Firing Rate",
    )
    if smooth_firing_rate:
        # Plot smoothed firing rate
        ax.plot(
            frames,
            cell_fr_smooth,
            linestyle="-",
            color="black",
            label="Smoothed Firing Rate",
        )
    if add_highlights:
        # Highlight the presence of the word
        ax.fill_between(
            frames,
            cell_fr_smooth,
            where=binary_mask > 0,
            color="red",
            alpha=0.3,
            label=f"Word '{word}' Present",
        )

    # Highlight the masked frames
    ax.fill_between(
        frames,
        ax.get_ylim()[1],
        where=~video_data.masked_frames_index,
        color="gray",
        alpha=0.5,
        label="Masked Frames",
    )

    # Customize plot
    ax.set_title(f"Neuron {cell_id} in {brain_area}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # Optionally plot box plot and violin plot
    if show_boxplot or show_violinplot:
        # Prepare data for plotting
        # Convert word presence to a categorical variable
        single_unit_df[f"{word}_present"] = single_unit_df[f"{word}_present"].fillna(0)
        single_unit_df[f"{word}_present"] = single_unit_df[f"{word}_present"].astype(
            int
        )
        single_unit_df = single_unit_df.dropna(subset=["firing_rate"])
        single_unit_df[f"{word}_present"] = single_unit_df[f"{word}_present"].map(
            {0: "Absent", 1: "Present"}
        )

        if show_boxplot:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(
                x=f"{word}_present", y="firing_rate", data=single_unit_df, ax=ax
            )
            ax.set_title(f"Firing Rate by Word Presence for Neuron {cell_id}")
            ax.set_xlabel(f"Word '{word}' Presence")
            ax.set_ylabel("Firing Rate (Hz)")
            plt.tight_layout()
            plt.show()

        if show_violinplot:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(
                x=f"{word}_present", y="firing_rate", data=single_unit_df, ax=ax
            )
            ax.set_title(
                f"Firing Rate Distribution by Word Presence for Neuron {cell_id}"
            )
            ax.set_xlabel(f"Word '{word}' Presence")
            ax.set_ylabel("Firing Rate (Hz)")
            plt.tight_layout()
            plt.show()

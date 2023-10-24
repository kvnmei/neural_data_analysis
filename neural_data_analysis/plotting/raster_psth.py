import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_raster_psth(spike_times, psth, time_window, bin_size, step_size, plot_params):
    """Plots a raster and PSTH for spikes to an event

    Args:
        spike_times (array):
        psth (array):
        time_window (list):
        bin_size (float):
        step_size (float):
        plot_params (dict):

    Returns:
        None
    """
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")

    # this code is to handle the case where there is only one event
    if not isinstance(spike_times[0], np.ndarray):
        spike_times = np.array([spike_times])

    total_events = len(spike_times)

    fig, ax = plt.subplots(
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

    ax[0].eventplot(spike_times, linewidths=0.5)
    ax[0].set_yticks(np.arange(total_events))
    ax[0].set_yticklabels(np.arange(1, total_events + 1))
    # ax[0].set_yticklabels(np.arange(1, total_events + 1)[::-1])
    ax[0].set_xlabel(plot_params.get("xlabel", "Time (s)"))
    ax[0].set_ylabel(plot_params.get("ylabel", "Event"))
    # ax[0].set_title(plot_params.get('title', "Raster Plot with PSTH"))

    ax[1].plot(
        np.arange(time_window[0], time_window[1] + step_size - bin_size, step_size)
        + bin_size / 2,
        psth,
        linewidth=1,
    )
    ax[1].set_xlim([time_window[0], time_window[1]])
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Firing rate (spikes/s)")
    # plot the stimulus time as a vertical line
    # ax[1].axvline(x=0, color="black", linestyle="--")

    for time in plot_params["scene_cut_times"]:
        ax[1].axvline(x=time, color="green", linestyle="-", linewidth=0.5)

    for time in plot_params["scene_change_times"]:
        ax[1].axvline(x=time, color="orange", linestyle="-", linewidth=0.75)

    plt.savefig(f"{plot_params['save_path']}", dpi=300, bbox_inches="tight")
    plt.close(fig)

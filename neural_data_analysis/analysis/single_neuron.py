"""
Functions for single neuron analysis.

"""
from typing import List

import numpy as np


def calc_firing_rates(
    event_times: List[float], spike_times: np.ndarray, window: List[float]
) -> np.ndarray:
    """
    Calculate the firing rate in a given window relative to an event.

    Args:
        event_times (list[float]): length n_events
            list of event times
        spike_times (numpy array): shape (n_neurons, n_spike_times)
            array of single neuron spike times in experiment time
        window (list): [start, end]
            start and end time for firing rate window, relative to event

    Returns:
        firing_rates (numpy array): shape (n_neurons, n_events)
            array of spike rates for each event

    Example:
        Calculate the firing rate between 0.2 and 1.2 s after image onset for each image onset in an experiment.
        firing_rates = calc_firing_rates(event_times, spike_times, window=[0.2, 1.2])
    """
    firing_rates = np.zeros(len(event_times))
    bin_size = window[1] - window[0]
    for i in np.arange(len(event_times)):
        event = event_times[i]
        start = event + window[0]
        end = event + window[1]
        spikes_subset = spike_times[(spike_times > start) & (spike_times < end)]
        firing_rates[i] = len(spikes_subset) / bin_size
    return firing_rates


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

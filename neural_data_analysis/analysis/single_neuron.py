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



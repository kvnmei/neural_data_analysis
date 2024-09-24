"""
Functions to analyze LFP data.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from mne.filter import create_filter


# noinspection PyTupleAssignmentBalance
def apply_notch_filter(signal, fs, notch_freq=60, quality_factor=30, backend="scipy"):
    """
    Apply a Notch filter to remove specific frequency from the signal.

    Args:
        signal (np.ndarray): 1D array of the signal to be filtered
        fs (float): sampling frequency
        notch_freq (float): frequency to be removed
        quality_factor (float): quality factor
        backend (str): 'scipy' or 'mne'

    Returns:
        filtered_signal (np.ndarray): 1D array of the filtered signal
    """
    if backend == "scipy":
        b, a = butter(2, [notch_freq - 0.5, notch_freq + 0.5], btype="bandstop", fs=fs)
        filtered_signal = filtfilt(b, a, signal)
    elif backend == "mne":
        filtered_signal = create_filter(
            signal,
            fs,
            l_freq=notch_freq - 0.5,
            h_freq=notch_freq + 0.5,
            method="iir",
            iir_params=dict(ftype="butter", order=2, output="sos"),
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return filtered_signal

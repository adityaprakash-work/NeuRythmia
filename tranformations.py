# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-19

# --Needed functionalities


# ---DEPENDENCIES---------------------------------------------------------------
import tftb.processing.cohen as tpc
import numpy as np
import matplotlib.pyplot as plt


# ---SPECTROGRAM----------------------------------------------------------------
# The classes having a __call__ method are meant to act as stateful functions
# to allow for repeated calls without the overhead of initialization of some of
# the variables and biolerplate code.
class CohenTftb:
    """
    Wrapper for tftb.processing.cohen.<tfr>

    Parameters
    ----------
    name : str
        Name of the tfr method to use. Must be one of
        ["Spectrogram", "PageRepresentation", "PseudoPageRepresentation",
        "MargenauHillDistribution", "PseudoMargenauHillDistribution",
        "WignerVilleDistribution", "PseudoWignerVilleDistribution"]
    sampling_rate : int
        Sampling frequency of the signal in Hz
    **kwargs : dict
        Keyword arguments to be passed to the tfr method

    Returns
    -------
    tfr : np.ndarray
        Time-frequency representation of the signal
    """

    SUPPORTED_COHEN_TFR = {
        "pr": tpc.PageRepresentation,
        "ppr": tpc.PseudoPageRepresentation,
        "mhd": tpc.MargenauHillDistribution,
        "wvd": tpc.WignerVilleDistribution,
        "spec": tpc.Spectrogram,
        "pmhd": tpc.PseudoMargenauHillDistribution,
        "pwvd": tpc.PseudoWignerVilleDistribution,
    }

    def __init__(self, name, sampling_rate, **kwargs):
        if name not in self.SUPPORTED_COHEN_TFR:
            raise ValueError(
                f"TFR must be one of {list(self.SUPPORTED_COHEN_TFR.keys())}"
            )
        else:
            self.name = name
        self.spec_method = self.SUPPORTED_COHEN_TFR[self.name]
        self.sampling_rate = sampling_rate
        self.time_stamps = None
        self.kwargs = kwargs

    def __call__(self, signal):
        if self.time_stamps is None:
            self.time_stamps = np.arange(signal.shape[0]) / self.sampling_rate
        tfr = np.empty((signal.shape[0], signal.shape[0] // 2, signal.shape[-1]))
        for channel in range(signal.shape[-1]):
            spec = self.spec_method(
                signal[:, channel], time_stamps=self.time_stamps, **self.kwargs
            )
            tfrc, _, _ = spec.run()
            tfrc = tfrc[: signal.shape[0] // 2]
            tfr[:, :, channel] = tfrc.T
        return tfr


def window_aggregate(arr, ax0idxs, ax1idxs):
    """
    Aggregate the values of a 2D array over a window of indices along the
    specified axes.

    Parameters
    ----------
    arr : np.ndarray
        Array to be aggregated
    ax0idxs : list of int
        Indices along the first axis to aggregate over
    ax1idxs : list of int
        Indices along the second axis to aggregate over

    Returns
    -------
    warr : np.ndarray
        Aggregated array
    """
    warr = np.empty((len(ax0idxs) + 1, len(ax1idxs) + 1, arr.shape[-1]))
    ax0idxs = [0] + ax0idxs + [None]
    ax1idxs = [0] + ax1idxs + [None]
    for i in range(len(ax0idxs) - 1):
        for j in range(len(ax1idxs) - 1):
            warr[i, j] = np.sum(
                arr[ax0idxs[i] : ax0idxs[i + 1], ax1idxs[j] : ax1idxs[j + 1]],
                axis=(0, 1),
            )
    return warr

# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-29

# --Needed functionalities
# 1. Using tf.data.Dataset.map to apply transformations to the dataset


# ---DEPENDENCIES---------------------------------------------------------------
import tftb.processing.cohen as tpc
import numpy as np
import mne


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
        self.kwargs = kwargs
        try:
            self.time_stamps = self.kwargs["time_stamps"]
        except:
            self.time_stamps = None

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


# ---PROCESSES------------------------------------------------------------------
class Process:
    def __init__(self, inner_process=None, paths=None, labels=None):
        self.inner_process = inner_process
        self.paths = paths
        self.labels = labels

    def __iter__(self):
        if self.inner_process is None:
            return self._consume_fl()
        else:
            return self._consume_ip()

    def __call__(self, inner_process=None, paths=None, labels=None):
        self.inner_process = inner_process
        self.paths = paths
        self.labels = labels
        return self

    def _consume_ip(self):
        for d, l in self.inner_process:
            try:
                d, l = self.transform(d, l)
                yield d, l
            except:
                pass

    def _consume_fl(self):
        for p, l in zip(self.paths, self.labels):
            try:
                d = self.load(p)
                d, l = self.transform(d, l)
                yield d, l
            except:
                pass

    def load(self, path):
        raise NotImplementedError

    def transform(self, data, label):
        raise NotImplementedError


class Default(Process):
    """
    Process to return the data and label as it is

    Returns
    -------
    process : Process
        Instance of the Process class
    """

    def load(self, path):
        return np.load(path)

    def transform(self, data, label):
        return data, label


class TimeSlice(Process):
    """
    Process to slice the data along the time axis using the start and end time
    in seconds if inner process has a parameter called 'sr' too, otherwise'sr'
    parameter is set to 1 and start and end times have to gives as indices.

    Parameters
    ----------
    start : int
        Start index of the slice
    end : int
        End index of the slice

    Returns
    -------
    process : Process
        Instance of the Process class
    """

    def __init__(self, start, end, sr=None):
        super().__init__()
        self.start = start
        self.end = end
        self.sr = sr

    def transform(self, data, label):
        ts = data.shape[0] / self.sr
        if self.start < ts and self.end < ts:
            data = data[self.start * self.sr : self.end * self.sr, ...]
            return data, label
        else:
            raise Exception

    def __call__(self, inner_process):
        self.inner_process = inner_process
        if self.sr is None:
            if inner_process.sr is not None or inner_process.sr != 1:
                self.sr = inner_process.sr
            else:
                self.sr = 1
        return self


class ChannelSelector(Process):
    """
    Process to select a subset of channels from the data

    Parameters
    ----------
    channels : list[int]
        List of indices of the channels to be selected

    Returns
    -------
    process : Process
        Instance of the Process class
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def transform(self, data, label):
        if data.shape[-1] >= len(self.channels):
            data = data[..., self.channels]
            return data, label
        else:
            raise Exception


class CohenTftbP(Process):
    """
    Process to compute the time-frequency representation of the data using
    tftb.processing.cohen.<tfr>

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
    process : Process
        Instance of the Process class
    """

    def __init__(self, name, sampling_rate=None, **kwargs):
        super().__init__()
        self.name = name
        self.sr = sampling_rate
        self.kwa = kwargs
        self.coh = CohenTftb(name=self.name, sampling_rate=self.sr, **self.kwa)

    def transform(self, data, label):
        try:
            data = self.coh(data)
            return data, label
        except:
            raise Exception

    def __call__(self, inner_process):
        self.inner_process = inner_process
        if self.sr is None:
            if inner_process.sr is not None or inner_process.sr != 1:
                self.sr = inner_process.sr
                self.coh.sampling_rate = self.sr
            else:
                raise ValueError("Invalid sampling rate, None and 1 reserved")
        return self


class LoadFif(Process):
    """
    Process to load fif files using mne.io.read_raw_fif()

    Parameters
    ----------
    paths : list[str]
        List of paths to the files to be loaded
    labels : list[str]
        List of labels corresponding to the files to be loaded

    Returns
    -------
    process : Process
        Instance of the Process class
    """

    def __init__(self, paths, labels):
        super().__init__(paths=paths, labels=labels)
        self.sr = int(self.load(paths[0]).info["sfreq"])

    def load(self, path):
        return mne.io.read_raw_fif(path, verbose="ERROR")

    def transform(self, data, label):
        data, _ = data[:, :]
        data = data.T
        return data, label

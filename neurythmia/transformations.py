# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-19

# --Needed functionalities


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


# ---PROCESSORS-----------------------------------------------------------------
class Processor:
    """
    Processor is a generator that encapsulates preprocessing transformations
    to aid in lazy-loading of a dataset. It is primarily meant to be used in
    conjunction with the NRCDataset class. The Processor class is meant to be
    inherited from and the child class must implement the transform() and load()
    methods.

    Processors can be chained together to form a preprocessing pipeline. While
    initializing a Processor, the first parameter can be another processor. The
    root of the chain mus be provided with the file_paths and file_labels.

    Stateful processor must be separately instantiated and only then placed in a
    processor chain as an object, and strictly not as a class definition pointer
    . Such processors must implement a __call__ method on a singular parameter
    called processor, which executes atleast the following code:

        def __call__(self, processor):
            self.processor = processor
            self.len = len(processor)
            return self

    The __call__ method is accessed with the same code as a new instiation i.e
    X(), where X is either an object or a class definition. The trick is that
    the first position is for a processor which makes the cosntructor need no
    further arguments.

    Parameters
    ----------
    processor : Processor
        Processor to be used as a generator. If None, the file_paths and
        file_labels parameters must be provided
    file_paths : list[str]
        List of paths to the files to be loaded
    file_labels : list[str]
        List of labels corresponding to the files to be loaded

    Returns
    -------
    processor : Processor
        Instance of the Processor class
    """

    def __init__(self, processor=None, file_paths=None, file_labels=None):
        self.processor = processor
        self.file_paths = file_paths
        self.file_labels = file_labels
        try:
            if self.processor is None:
                self.len = len(file_paths)
            else:
                self.len = len(self.processor)
        except:
            self.len = None
        self.index = 0

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.len:
            raise StopIteration
        if self.processor is None:
            fp = self.file_paths[self.index]
            fl = self.file_labels[self.index]
            data = self.load(fp)
            data, fl = self.transform(data, fl)
            self.index += 1
            return data, fl
        else:
            data, fl = next(self.processor)
            data, fl = self.transform(data, fl)
            self.index += 1
            return data, fl

    # For stateful processors
    def __call__(self, processor):
        self.processor = processor
        self.len = len(processor)
        return self

    # Following two methods have to be implemented by the child class
    def transform(self, data, label):
        raise NotImplementedError

    def load(self, file_path):
        raise NotImplementedError


class LoadFif(Processor):
    """
    Processor to load fif files using mne.io.read_raw_fif()

    Parameters
    ----------
    file_paths : list[str]
        List of paths to the files to be loaded
    file_labels : list[str]
        List of labels corresponding to the files to be loaded

    Returns
    -------
    processor : Processor
        Instance of the Processor class
    """

    # Defining a new __init__ for sr and making parameters compulsory
    def __init__(self, file_paths, file_labels):
        super().__init__(file_paths=file_paths, file_labels=file_labels)
        self.sr = int(self.load(file_paths[0]).info["sfreq"])

    def load(self, file_path):
        return mne.io.read_raw_fif(file_path, verbose="ERROR")

    def transform(self, data, label):
        data, _ = data[:, :]
        data = data.T
        return data, label


class TimeSlice(Processor):
    """
    Processor to slice the data along the time axis using the start and end time
    in seconds if inner processor has a parameter called 'sr' too, otherwise
    'sr' parameter is set to 1 and start and end times have to gives as indices.

    Parameters
    ----------
    start : int
        Start index of the slice
    end : int
        End index of the slice

    Returns
    -------
    processor : Processor
        Instance of the Processor class
    """

    def __init__(self, start, end, sr=None):
        super().__init__()
        self.start = start
        self.end = end
        self.sr = sr

    def transform(self, data, label):
        data = data[self.start * self.sr : self.end * self.sr, ...]
        return data, label

    def __call__(self, processor):
        self.processor = processor
        self.len = len(processor)
        if self.sr is None:
            if processor.sr is not None or processor.sr != 1:
                self.sr = processor.sr
            else:
                self.sr = 1
        return self


class ChannelSelector(Processor):
    """
    Processor to select a subset of channels from the data

    Parameters
    ----------
    channels : list[int]
        List of indices of the channels to be selected

    Returns
    -------
    processor : Processor
        Instance of the Processor class
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def transform(self, data, label):
        data = data[..., self.channels]
        return data, label


class CohenTftbP(Processor):
    """
    Processor to compute the time-frequency representation of the data using
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
    processor : Processor
        Instance of the Processor class
    """

    def __init__(self, name, sampling_rate=None, **kwargs):
        super().__init__()
        self.name = name
        self.sr = sampling_rate
        self.kwa = kwargs
        self.coh = CohenTftb(name=self.name, sampling_rate=self.sr, **self.kwa)

    def transform(self, data, label):
        data = self.coh(data)
        return data, label

    def __call__(self, processor):
        self.processor = processor
        self.len = len(processor)
        if self.sr is None:
            if processor.sr is not None or processor.sr != 1:
                self.sr = processor.sr
            else:
                raise ValueError("Invalid sampling rate, None and 1 reserved")
        return self

# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-29

# --Needed functionalities
# 1. Using tf.data.Dataset.map to apply transformations to the dataset


# ---DEPENDENCIES---------------------------------------------------------------
import tftb.processing.cohen as tpc
import numpy as np
import mne
import tensorflow as tf


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


# ---PROCESS CHAIN--------------------------------------------------------------
class ProcessChain:
    """
    Class to create a process chain

    Parameters
    ----------
    processes : list[Process]
        List of processes to be chained

    Returns
    -------
    process_chain : ProcessChain
        Instance of the ProcessChain class
    """

    CONNECT_METHODS = [
        "clubbed_map_transforms",
        "sequential_map_transforms",
        "from_generators",
    ]

    def __init__(self, connect_method, processes):
        if connect_method not in self.CONNECT_METHODS:
            raise ValueError(f"Connect method not in {self.CONNECT_METHODS}")
        self.connect_method = connect_method
        self.processes = processes
        for process in self.processes:
            if self.connect_method not in process.fccm:
                if self.connect_method not in process.acmh:
                    raise ValueError(
                        f"Connect method not compatible with {process.__name__}"
                    )

    @staticmethod
    def chaining_asset_0(processes, rooted=True):
        def cpt(x, y):
            # if rooted, x, must be path-like
            if rooted:
                x, y = processes[0].load(x), y
            # Even root process transform has to be called so no [1:]
            for process in processes:
                x, y = process.transform(x, y)
            return x, y

        return cpt

    @staticmethod
    def chaining_asset_1(processes, paths=None, labels=None, rooted=True):
        if rooted:
            cp = processes[0](paths=paths, labels=labels)
        else:
            cp = processes[0]
        for process in processes[1:]:
            cp = process(cp)
        return cp

    @staticmethod
    def chaining_asset_2(D, connect_method, processes, rooted=True):
        def load(x, y):
            x, y = processes[0].load(x), y
            x, y = processes[0].transform(x, y)
            return x, y

        if rooted:
            D = D.map(
                lambda x, y: tf.py_function(
                    load, inp=[x, y], Tout=[tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
        for process in processes[1:]:
            if connect_method in process.acmh:
                D = process.handler(connect_method=connect_method, D=D)
            else:
                D = D.map(
                    lambda x, y: tf.py_function(
                        process.transform,
                        inp=[x, y],
                        Tout=[tf.float32, tf.float32],
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
        return D

    def apply(self, D=None, paths=None, labels=None):
        if self.connect_method == "clubbed_map_transforms":
            rooted = False
            if D is None:
                D = tf.data.Dataset.from_tensor_slices((paths, labels))
                rooted = True
            cpt = self.chaining_asset_0(self.processes, rooted=rooted)
            D = D.map(
                lambda x, y: tf.py_function(
                    cpt, inp=[x, y], Tout=[tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )

        elif self.connect_method == "sequential_map_transforms":
            rooted = False
            if D is None:
                D = tf.data.Dataset.from_tensor_slices((paths, labels))
                rooted = True
            D = self.chaining_asset_2(
                D,
                "sequential_map_transforms",
                self.processes,
                rooted=rooted,
            )

        elif self.connect_method == "from_generators":
            rooted = False
            if D is None:
                rooted = True
            cp = self.chaining_asset_1(
                self.processes, paths=paths, labels=labels, rooted=rooted
            )
            D = tf.data.Dataset.from_generator(
                lambda: cp,
                (tf.float32, tf.float32),
            )

        # To filter out Exceptions in absence of the escaping yield trick
        # of Processes. This functionality is not removed from Processes
        # because they are used in writing too, where tf.data.Dataset is
        # not prepared
        return D.ignore_errors()


# ---PROCESSES------------------------------------------------------------------
class Process:
    """
    Base class for all the processes

    Parameters
    ----------
    inner_process : Process
        Inner process to be called
    paths : list[str]
        List of paths to the files to be loaded
    labels : list[str]
        List of labels corresponding to the files to be loaded
    compatible_connect_methods : list[str]
        List of connect methods compatible with the process

    Returns
    -------
    process : Process
        Instance of the Process class

    Notes
    -----
    1. While subsclassing the Process class, the subclass must implement the
       load and transform methods as per requirement.
    2. The Process class can be used as a decorator to create a process chain
       using the __call__ method.
    3. The Process class can be used as an iterator to create a process chain
       using the __iter__ method.
    4. The Process class can be used as a callable to create a process chain
       using the __call__ method.

    Process Chain
    -------------
    A process chain is a sequence of processes that are applied to the data
    one after the other. The process chain can be created as any iterator of
    Process definitions where state is not needed and instances of Process in
    case state is needed.
    It is recommended that load() and transform() be defined as static-methods
    of the Processes that are do not need access to 'self' (stateless) to
    perform those operations. This is to ensure the callabalility of those
    methods, when they are referenced as class definitions.

    Handlers
    ---------
    Handlers are functions that are used modify the chaining of processes if
    needed by ProcessChain. It is recommended to implement handlers with a
    single positional argument, the intended connect method and **kwargs to
    allow for flexibility in using the handler according to it. If a handler for
    a particular connect method is implemented, then it is advised to not add
    the connect method to the compatible_connect_methods list.
    """

    # fully compatible connect methods
    fccm = ProcessChain.CONNECT_METHODS
    # alternate connect methods handlers
    acmh = []

    def __init__(self, inner_process=None, paths=None, labels=None):
        self.inner_process = inner_process
        self.paths = paths
        self.labels = labels

    def __name__(self):
        return self.__class__.__name__

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

    @staticmethod
    def path_decode(load):
        def wrapper(path):
            if type(path) is not str:
                path = path.numpy().decode()
            return load(path)

        return wrapper

    def load(self, path):
        raise NotImplementedError

    def transform(self, data, label):
        raise NotImplementedError

    def handler(self, connect_method, **kwargs):
        raise NotImplementedError


class Default(Process):
    """
    Process to return the data and label as it is

    Returns
    -------
    process : Process
        Instance of the Process class
    """

    @staticmethod
    @Process.path_decode
    def load(path):
        return np.load(path)

    @staticmethod
    def transform(data, label):
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


class TimePleat(TimeSlice):
    """
    Process to 'pleat' data across time axis, resulting in multiple segments
    being drawn from a single time series.

    Parameters
    ----------
    start : int
        Start index of the slice
    segment_length : int
        Length of the segment in seconds
    overlap : int
        Overlap between consecutive segments in seconds
    end : int
        End index of the slice

    Returns
    -------
    process : Process
        Instance of the Process class
    """

    fccm = ["from_generators"]
    acmh = ["sequential_map_transforms"]

    def __init__(self, start, segment_length, overlap=0, end=None, sr=None):
        super().__init__(start, end, sr)
        self.sl = segment_length
        self.overlap = overlap

    def _consume_ip(self):
        for d, l in self.inner_process:
            try:
                pd, pl = self.transform(d, l)
                for seg, lb in zip(pd, pl):
                    yield seg, lb
            except:
                pass

    def transform(self, data, label):
        ts = data.shape[0] / self.sr
        step = self.sl - self.overlap
        if self.end <= ts and self.end - self.start >= self.sl:
            _data = []
            for ss in range(self.start, self.end, step):
                se = ss + self.sl
                if se <= self.end:
                    _data.append(data[ss * self.sr : se * self.sr, ...])
            data = np.array(_data)
            label = [label] * data.shape[0]
            return data, label
        else:
            raise Exception

    def handler(self, connect_method, **kwargs):
        if connect_method == "sequential_map_transforms":
            D = kwargs["D"]
            D = D.map(
                lambda x, y: tf.py_function(
                    self.transform, inp=[x, y], Tout=[tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            D = D.unbatch()
            return D


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
            try:
                data = data[..., self.channels]
            except:
                data = tf.gather(data, self.channels, axis=-1)
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

    @staticmethod
    @Process.path_decode
    def load(path):
        return mne.io.read_raw_fif(path, verbose="ERROR")

    @staticmethod
    def transform(data, label):
        data, _ = data[:, :]
        data = data.T
        return data, label

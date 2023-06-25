# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-22

# --Needed functionalities
# 1. Integration with tf.data.Dataset
# 2. Integration of loaders with NRM.json
# 3. Refinement of NRMD.json and writers
# 4. Constructing a NeuRythmia Dataset class with integrated features
# 5. Deprecating separate loaders and writers


# ---DEPENDENCIES---------------------------------------------------------------
import os
from typing import Any, Iterable, Callable
from os.path import join as opj
from os.path import exists as ope
import glob
import json
from tqdm import tqdm

import mne
import numpy as np
import scipy as sp
import tftb
import tensorflow as tf

from . import utils


# ---LOAD BASES-----------------------------------------------------------------
class LoadBase1(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        directory=None,
        file_type=None,
        classes=None,
        channels=None,
        shuffle=True,
        validation_split=None,
        validation_batch_size=None,
        seed=None,
        file_info=None,
    ):
        self.batch_size = batch_size
        self.directory = directory
        self.file_type = file_type
        self.classes = classes
        self.channels = channels
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        if file_info is None:
            if self.directory is None:
                raise ValueError(
                    "Directory must be specified if file_info is not given"
                )
            self.file_info = [
                [path, c]
                for c, cln in enumerate(self.classes)
                for path in glob.glob(
                    opj(self.directory, cln, f'*.{self.file_type or "*"}')
                )
            ]
        else:
            self.file_info = file_info
        self.file_info.sort(key=lambda x: x[0])
        self.file_info = np.array(self.file_info)
        self.rng.shuffle(self.file_info)

        if self.validation_split is not None:
            if validation_batch_size is None:
                self.validation_batch_size = batch_size
            else:
                self.validation_batch_size = validation_batch_size
            self.pri_file_info = self.rng.choice(
                self.file_info,
                size=int(len(self.file_info) * (1 - self.validation_split)),
                replace=False,
            )
            self.aux_file_info = self.file_info[
                ~np.isin(self.file_info, self.pri_file_info)
            ]
            self.aux_dgen = LoadBase1(
                data_shape=self.data_shape,
                batch_size=self.validation_batch_size,
                classes=self.classes,
                shuffle=self.shuffle,
                seed=self.seed,
                file_info=self.aux_file_info,
            )
        else:
            self.pri_file_info = self.file_info
            self.aux_file_info = None
            self.aux_dgen = None

        self.on_epoch_end()
        self.data_shape = self._processing(self.pri_file_info[0][0]).shape

    def __len__(self):
        return len(self.pri_file_info) // self.batch_size

    def __getitem__(self, index):
        batch_info = self.pri_file_info[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        Xb, Yb = self._batch_generator(batch_info)
        return Xb, Yb

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pri_file_info)

    def _batch_generator(self, batch_info):
        Xb = np.empty((self.batch_size, *self.data_shape))
        Yb = np.empty((self.batch_size), dtype=int)
        for i, (f, c) in enumerate(batch_info):
            Xb[i], Yb[i] = self._processing(f), c
        return Xb, Yb

    # This method should be overriden by the child class and should include code
    # for loading, processing, and returning a single data sample
    def _processing(self, f):
        raise NotImplementedError


# ---EEG------------------------------------------------------------------------
class EEGRaw(LoadBase1):
    supported_types = ["nrraw"]

    def __init__(
        self,
        batch_size,
        directory=None,
        file_type="nrraw",
        classes=None,
        channels=None,
        shuffle=True,
        validation_split=None,
        validation_batch_size=None,
        seed=None,
        file_info=None,
    ):
        # Checking whether the file_type is valid. Using 'self' instead of
        # 'EEGRaw' for allowing other types at instance level that can be
        # handled by sophisticated '_processing'(s)
        if file_type not in self.supported_types:
            raise ValueError(f"File type must be one of {self.supported_types}")

        # Calling the parent class constructor
        super().__init__(
            batch_size=batch_size,
            directory=directory,
            file_type=file_type,
            classes=classes,
            channels=channels,
            shuffle=shuffle,
            validation_split=validation_split,
            validation_batch_size=validation_batch_size,
            seed=seed,
            file_info=file_info,
        )

    # Normalizing across all channels
    def _processing(self, f):
        x = np.load(f)[..., self.channels]
        x = (x - x.mean()) / x.std()
        return x


class EEGSpectrogram(LoadBase1):
    supported_types = ["nrspec", "nrraw"]

    def __init__(
        self,
        batch_size,
        directory=None,
        file_type="nrspec",
        classes=None,
        channels=None,
        shuffle=True,
        validation_split=None,
        validation_batch_size=None,
        seed=None,
        file_info=None,
        temporal=False,
        spec_transform=None,
    ):
        # Check whether the file_type is valid. Using 'self' instead of
        # 'EEGSpectrogram' for allowing other types at instance level that
        # can be handled by sophisticated '_processing'(s)
        if file_type not in self.supported_types:
            raise ValueError(f"File type must be one of {self.supported_types}")

        # Call the parent class constructor
        super().__init__(
            batch_size=batch_size,
            directory=directory,
            file_type=file_type,
            classes=classes,
            channels=channels,
            shuffle=shuffle,
            validation_split=validation_split,
            validation_batch_size=validation_batch_size,
            seed=seed,
            file_info=file_info,
        )
        self.temporal = temporal
        self.spec_transform = spec_transform

    # spec_transform should work for multi-channel data sample
    def _processing(self, f):
        x = np.load(f)[..., self.channels]
        x = (x - x.mean()) / x.std()
        if self.spec_transform is not None:
            x = self.spec_transform(x)
        if self.temporal:
            x = x.reshape(x.shape[0], -1)
        return x

    def __init__(
        self,
        dataset_name,
        classes,
        base_dir,
        register=False,
        s_time=0,
        e_time=20,
    ):
        super().__init__(
            dataset_name=dataset_name,
            classes=classes,
            file_type="nrraw",
            base_dir=base_dir,
            register=register,
        )
        self.s_time = s_time
        self.e_time = e_time
        self.segment_length = int((self.e_time - self.s_time))

    def _processing(self, f):
        eegep = mne.io.read_raw_fif(f, verbose="ERROR")
        sr = int(eegep.info["sfreq"])
        data, _ = eegep[:, :]
        segment = data[:, self.s_time * sr : self.e_time * sr].T
        return segment


# ---NRDataset------------------------------------------------------------------
class NRCDataset:
    def __init__(self, base_dir, dataset_name):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.path = opj(self.base_dir, self.dataset_name)

        # flags
        self._de = False  # Dataset exists
        self._me = False  # metadata exists
        self._cc = False  # can create

        if ope(self.path):
            self._de = True
            if ope(opj(self.path, "nrcm.json")):
                self.metadata = utils.NRCM(path=opj(self.path, "nrcm.json"))
                self._me = True
                print(f"NR > Registered dataset {self.dataset_name} detected")
            else:
                print("NR > Unregistered dataset detected")
        else:
            self._cc = True
            print("NR > No dataset found, create new")

        self.D = None  # td.data.Dataset

    def create(self, classes=None, file_type=None, data_shape=None):
        if self._cc == True:
            for cl in classes:
                os.makedirs(opj(self.base_dir, self.dataset_name, cl))
            self.metadata = utils.NRCM(
                dataset_name=self.dataset_name,
                classes=classes,
                file_type=file_type,
                data_shape=data_shape,
            )
            self.metadata.save(opj(self.path, "nrcm.json"))
            self._cc = False
            self._me = True
            print(f"NR > Dataset created at {self.path}")
        else:
            raise ValueError("create(): called on an existing dataset")

    def write(self, file_name, tag, data=None):
        if self._me == False:
            raise ValueError("write(): called on ambiguous dataset")
        if file_name in self.metadata.nrm["files"]:
            if tag in self.metadata.nrm["files"][file_name]:
                # Overwrite not allowed
                raise ValueError(f"'{file_name}' already has tag '{tag}'")
        else:
            if tag in self.metadata.nrm["classes"]:
                if data is not None:
                    np.save(
                        opj(
                            self.base_dir,
                            self.dataset_name,
                            tag,
                            file_name,
                        ),
                        data,
                    )
                else:
                    raise ValueError("NoneType data cannot be written")
                self.metadata.nrm["size"][tag] += 1
            else:
                # First tag should be the class label
                raise ValueError(
                    f"First tag should be one of {self.metadata.nrm['classes']}"
                )
        self.metadata.add(name=file_name, tag=tag)

    def erase(self, file_name, tag=None, total_erase=False):
        if self._me == False:
            raise ValueError("erase(): called on ambiguous dataset")
        if file_name in self.metadata.nrm["files"]:
            if total_erase:
                cl = self.metadata.nrm["files"][file_name][0]
                os.remove(
                    opj(
                        self.base_dir,
                        self.dataset_name,
                        cl,  # class name
                        file_name + ".npy",
                    )
                )
                self.metadata.remove(name=file_name, total_remove=True)
                self.metadata.nrm["size"][cl] -= 1
            else:
                if tag in self.metadata.nrm["files"][file_name]:
                    if tag not in self.metadata.nrm["classes"]:
                        self.metadata.remove(name=file_name, tag=tag)
                    else:
                        raise ValueError(f"Class tag cannot be erased")
                else:
                    raise ValueError(
                        f"File '{file_name}' does not have tag '{tag}' to erase"
                    )
        else:
            raise ValueError(f"File {file_name} does not exist")

    def register(self, file_type, ext="npy"):
        dsr = False
        if self._de == True:
            if self._me == False:
                ld = os.listdir(self.path)
                classes = [i for i in ld if os.path.isdir(opj(self.path, i))]
                self.metadata = utils.NRCM(
                    dataset_name=self.dataset_name,
                    classes=classes,
                    file_type=file_type,
                )
                for fp in tqdm(glob.glob(opj(self.path, "**", f"*.{ext}"))):
                    if drc == False:
                        d = np.load(fp)
                        self.metadata.nrm["data_shape"] = d.shape
                        drc = True
                    fn = os.path.basename(fp).split(".")[0]
                    cn = os.path.basename(os.path.dirname(fp))
                    self.metadata.add(name=fn, tag=cn)
                    self.metadata.nrm["size"][cn] += 1
                self.metadata.save(opj(self.path, "nrcm.json"))
                self._me = True
                print(f"NR > Dataset {self.dataset_name} registered")
            else:
                raise ValueError("register(): called on registered dataset")
        else:
            raise ValueError("register(): called on non-existent dataset")

    def _default_processor(self, paths, labels):
        for path, label in zip(paths, labels):
            yield np.load(path), label

    def connect(
        self,
        tag_combinations=None,
        batch_size=8,
        processor=None,
        categorical=True,
        enforce_binary=False,
    ):
        if self._de == False:
            raise ValueError("connect(): called on non-existent dataset")
        if self._me == False:
            raise ValueError("connect(): called without nrcm.json")
        self.D = None  # Dataset set to None
        if tag_combinations is None:
            tag_combinations = [[cln] for cln in self.metadata.nrm["classes"]]
        if processor is None:
            processor = self._default_processor

        fns = self.metadata.fetch(tag_combinations)
        fls = [self.metadata.nrm["files"][fn][0] for fn in fns]
        fps = [opj(self.path, cln, fn + ".npy") for cln, fn in zip(fls, fns)]
        ils = [self.metadata.nrm["classes"].index(cln) for cln in fls]
        n_classes = len(np.unique(ils))
        if enforce_binary:
            if n_classes != 2:
                raise ValueError("Binary label cannot be enforced, classes > 2")
            else:
                ils = np.array(ils) // np.array(ils).max()
        else:
            if categorical:
                ils = tf.keras.utils.to_categorical(ils, num_classes=n_classes)
            else:
                ils = np.array(ils)
        self.D = tf.data.Dataset.from_generator(
            lambda: processor(fps, ils), (tf.float32, tf.float32)
        )
        self.D.shuffle(len(fps))
        self.D = self.D.batch(batch_size=batch_size)
        print(f"NR > Connected to dataset {self.dataset_name}")

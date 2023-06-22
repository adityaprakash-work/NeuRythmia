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
from os.path import join as opj
import glob
import json
import tqdm

import mne
import numpy as np
import scipy as sp
import tftb
import tensorflow as tf

import utils


# ---LOAD BASES-----------------------------------------------------------------
class LoadBase1(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        directory=None,
        file_extension=None,
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
        self.file_extension = file_extension
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
                    opj(self.directory, cln, f'*.{self.file_extension or "*"}')
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
    supported_extensions = ["nrraw"]

    def __init__(
        self,
        batch_size,
        directory=None,
        file_extension="nrraw",
        classes=None,
        channels=None,
        shuffle=True,
        validation_split=None,
        validation_batch_size=None,
        seed=None,
        file_info=None,
    ):
        # Checking whether the file_extension is valid. Using 'self' instead of
        # 'EEGRaw' for allowing other extensions at instance level that can be
        # handled by sophisticated '_processing'(s)
        if file_extension not in self.supported_extensions:
            raise ValueError(
                f"File extension must be one of {self.supported_extensions}"
            )

        # Calling the parent class constructor
        super().__init__(
            batch_size=batch_size,
            directory=directory,
            file_extension=file_extension,
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
    supported_extensions = ["nrspec", "nrraw"]

    def __init__(
        self,
        batch_size,
        directory=None,
        file_extension="nrspec",
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
        # Check whether the file_extension is valid. Using 'self' instead of
        # 'EEGSpectrogram' for allowing other extensions at instance level that
        # can be handled by sophisticated '_processing'(s)
        if file_extension not in self.supported_extensions:
            raise ValueError(
                f"File extension must be one of {self.supported_extensions}"
            )

        # Call the parent class constructor
        super().__init__(
            batch_size=batch_size,
            directory=directory,
            file_extension=file_extension,
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


# ---WRITE BASES----------------------------------------------------------------
class WriteBase1:
    def __init__(
        self,
        dataset_name=None,
        classes=None,
        file_extension=None,
        base_dir=None,
        register=False,
    ):
        self.dataset_name = dataset_name
        self.classes = classes
        self.file_extension = file_extension
        self.base_dir = base_dir
        self.register = register

        if os.path.exists(opj(self.base_dir, dataset_name)):
            print(f"NR >> Dataset '{self.dataset_name}' already exists")
            try:
                self.metadata = utils.NRMDataset(
                    path=opj(self.base_dir, self.dataset_name, "NRMD.json")
                )
                self.metadata.info()
            except:
                if register:
                    self.metadata = utils.NRMDataset(
                        file_extension=self.file_extension,
                        dataset_name=self.dataset_name,
                        classes=self.classes,
                    )
                    for cl in self.classes:
                        os.makedirs(
                            opj(self.base_dir, self.dataset_name, cl),
                            exist_ok=True,
                        )
                        for f in glob.glob(
                            opj(
                                self.base_dir,
                                self.dataset_name,
                                cl,
                                f"*.{self.file_extension}",
                            )
                        ):
                            self.metadata.add(
                                name=os.path.basename(f).split(".")[0],
                                tag=cl,
                            )
                    self.metadata.update_size()
                    self.metadata.save(
                        opj(self.base_dir, self.dataset_name, "NRMD.json")
                    )
                else:
                    raise ValueError("NR >> Missing NRMD.json")
        else:
            print(f"NR >> Creating Dataset: '{self.dataset_name}'")
            for cl in self.classes:
                os.makedirs(opj(self.base_dir, self.dataset_name, cl))
            self.metadata = utils.NRMDataset(
                file_extension=self.file_extension,
                dataset_name=self.dataset_name,
                classes=self.classes,
            )

    def write(self, file_name, tag, data=None):
        if file_name in self.metadata.nrm["files"]:
            if tag in self.metadata.nrm["files"][file_name]:
                # Overwrite not allowed
                raise ValueError(f"File '{file_name}' already has tag <{tag}>")
        else:
            if tag in self.metadata.nrm["classes"]:
                np.save(
                    opj(
                        self.base_dir,
                        self.dataset_name,
                        tag,
                        f"{file_name}.{self.metadata.nrm['file_extension']}",
                    ),
                    data,
                )
            else:
                # First tag should be the class label
                raise ValueError(
                    f"First tag should be one of {self.metadata.nrm['classes']}"
                )
        self.metadata.add(name=file_name, tag=tag)

    def erase(self, file_name, tag=None, total_erase=False):
        if file_name in self.metadata.nrm["files"]:
            if total_erase:
                if (
                    tag in self.metadata.nrm["files"][file_name]
                    and tag in self.metadata.nrm["classes"]
                ):
                    os.remove(
                        opj(
                            self.base_dir,
                            self.dataset_name,
                            tag,
                            f"{file_name}.{self.metadata.nrm['file_extension']}",
                        )
                    )
                    self.metadata.remove(name=file_name, total_remove=True)
            else:
                if tag in self.metadata.nrm["files"][file_name]:
                    self.metadata.remove(name=file_name, tag=tag)
                else:
                    raise ValueError(
                        f"File '{file_name}' does not have tag <{tag}> to erase"
                    )
        else:
            raise ValueError(f"File {file_name} does not exist")

    def assimilate(self, path_to_nrd, file_extension, suffix):
        if os.path.exists(path_to_nrd):
            for cl in self.classes:
                print(f"NR >> Assimilating class: {cl} from {os.basename(path_to_nrd)}")
                for f in tqdm.tqdm(
                    glob.glob(opj(path_to_nrd, cl, f"*.{file_extension}"))
                ):
                    try:
                        d = self._processing(f)
                        self.write(
                            file_name=os.basename(f).split(".")[0] + suffix,
                            tag=cl,
                            data=d,
                        )
                    except:
                        print(f"Failed to assimilate {f}")
            self.metadata.update_size()
            self.meradata.save(opj(self.base_dir, self.dataset_name, "NRMD.json"))
        else:
            raise ValueError(f"Path '{path_to_nrd}' does not exist")

    def _processing(self, f):
        raise NotImplementedError


# ---EEG------------------------------------------------------------------------
class EEGRawWriter(WriteBase1):
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
            file_extension="nrraw",
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

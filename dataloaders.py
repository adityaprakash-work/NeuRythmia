# ---DEPENDENCIES---------------------------------------------------------------
import os
import glob

import mne
import numpy as np
import scipy as sp
import tftb
import tensorflow as tf


# ---BASES----------------------------------------------------------------------
class Base1(tf.keras.utils.Sequence):
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
                for c, class_name in enumerate(self.classes)
                for path in glob.glob(
                    os.path.join(
                        self.directory, class_name, f'*.{self.file_extension or "*"}'
                    )
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
            self.aux_dgen = Base1(
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
        self.data_shape = self._processing_function(self.pri_file_info[0][0]).shape

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
            Xb[i], Yb[i] = self._processing_function(f), c
        return Xb, Yb

    # This method should be overriden by the child class and should include code
    # for loading, processing, and returning a single data sample
    def _processing_function(self, f):
        pass


class EEGRaw(Base1):
    valid_extensions = ["nrraw"]

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
        # Checking whether the file_extension is valid
        # Using 'self' instead of 'EEGRaw' for allowing other extensions at instance
        # level that can be handled by sophisticated '_processing_function'(s)
        if file_extension not in self.valid_extensions:
            raise ValueError(
                f"Invalid file extension. Must be one of {self.valid_extensions}"
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
    def _processing_function(self, f):
        x = np.load(f)[..., self.channels]
        x = (x - x.mean()) / x.std()
        return x


class EEGSpectrogram(Base1):
    valid_extensions = ["nrspec", "nrraw"]

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
        # Check whether the file_extension is valid
        # Using 'self' instead of 'EEGSpectrogram' for allowing other extensions at
        # instance level that can be handled by sophisticated '_processing_function'(s)
        if file_extension not in self.valid_extensions:
            raise ValueError(
                f"Invalid file extension. Must be one of {self.valid_extensions}"
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

    def _processing_function(self, f):
        x = np.load(f)[..., self.channels]
        x = (x - x.mean()) / x.std()
        if self.spec_transform is not None:
            x = self.spec_transform(x)
        if self.temporal:
            x = x.reshape(x.shape[0], -1)
        return x

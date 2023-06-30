# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-29

# --Needed functionalities
# 1. More intuitive alternative of tag_combinations in NRCDataset required.


# ---DEPENDENCIES---------------------------------------------------------------
import os
from os.path import join as opj
from os.path import exists as ope
import glob
from tqdm import tqdm

import mne
import numpy as np
import tensorflow as tf

from . import utils
from . import transformations as tfm


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


# ---NRDataset------------------------------------------------------------------
class NRCDataset:
    """
    Class to interact with a dataset. The dataset is stored in a directory with
    the following structure:
    <dataset_name>
    ├── <nrcm.json>
    ├── <class_name_1>
    │   ├── <file_name_1>.<file_type>
    │   ├── <file_name_2>.<file_type>
    │   └── ...
    ├── <class_name_2>
    │   ├── <file_name_1>.<file_type>
    │   ├── <file_name_2>.<file_type>
    │   └── ...
    └── ...

    Parameters
    ----------
    base_dir : str
        Path to the directory containing the dataset
    dataset_name : str
        Name of the dataset

    Returns
    -------
    nrcd : NRCDataset
        Instance of the NRCDataset class

    Methods
    -------
    create
        Creates a new dataset
    write
        Writes data to the dataset
    erase
        Erases data from the dataset
    register
        Registers the dataset
    connect
        Connects to the dataset and returns a tf.data.Dataset object
    processed_write
        Writes processed data to the dataset sourced from another NRCDataset or
        explicit file paths and file labels.

    Usage
    -----
    For creating a new dataset:
    >>> import neurythmia as nr
    >>> nrcd = nr.NRCDataset(base_dir, dataset_name)
    >>> nrcd.create(classes, file_type, ext, data_shape)

    For writing data to the dataset:
    1. Sparse writing
    >>> nrcd.write(file_name, tag, data, bulk=False)
    2. Bulk writing, followed by manual saving
    >>> nrcd.write(file_name, tag, data, bulk=True)
    >>> nrcd.metadata.save(opj(nrcd.path, "nrcm.json"))

    For erasing data from the dataset:
    1. Sparse erasing
    >>> nrcd.erase(file_name, tag, total_erase=False, bulk=False)
    2. Bulk erasing, followed by manual saving
    >>> nrcd.erase(file_name, tag, total_erase=False, bulk=True)
    >>> nrcd.metadata.save(opj(nrcd.path, "nrcm.json"))

    For registering the dataset:
    >>> nrcd.register(file_type, ext, data_shape)

    For connecting to the dataset:
    >>> nrcd.connect(
            tag_combinations,
            batch_size,
            shuffle,
            process_chain,
            categorical,
            enforce_binary,
            extn
        )

    For processing and writing data to the dataset:
    >>> nrcd.processed_write(
            dpath,
            dtag_combinations,
            file_paths,
            alt_names,
            class_labels,
            process_chain
        )
    """

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

        self._default_process = tfm.Default

    def create(self, classes=None, file_type=None, ext="npy", data_shape=None):
        """
        Creates a new dataset

        Parameters
        ----------
        classes : list of str, optional
            List of class names. If None, the class names are inferred from the
            file names. Default is None.
        file_type : str, optional
            Type of the files in the dataset. If None, the file type is inferred
            from the file extension. Default is None.
        ext : str, optional
            Extension of the files in the dataset. Default is 'npy'.
        data_shape : tuple of int, optional
            Shape of the data in the dataset. If None, the shape is inferred
            from the first file. Default is None.
        """

        if self._cc == True:
            for cl in classes:
                os.makedirs(opj(self.base_dir, self.dataset_name, cl))
            self.metadata = utils.NRCM(
                dataset_name=self.dataset_name,
                classes=classes,
                file_type=file_type,
                file_ext=ext,
                data_shape=data_shape,
            )
            self.metadata.save(opj(self.path, "nrcm.json"))
            self._cc = False
            self._me = True
            print(f"NR > Dataset created at {self.path}")
        else:
            raise ValueError("create(): called on an existing dataset")

    def write(self, file_name, tag, data=None, bulk=False, save_method=np.save):
        """
        Writes data to the dataset

        Parameters
        ----------
        file_name : str
            Name of the file to be written
        tag : str
            Tag of the file to be written
        data : np.ndarray, optional
            Data to be written. If None, the file is created with the given
            name and tag. Default is None.
        bulk : bool, optional
            Whether to save the metadata after writing. Default is False.
        save_method : function, optional
            Method to be used for saving the data. Default is np.save.
            The method given should take two arguments: file path and data.
            A custom wrapper is recommended for methods that are of different
            format, which can then be used by passing the wrapper as the
            save_method argument.
        """

        if self._me == False:
            raise ValueError("write(): called on ambiguous dataset")
        if file_name in self.metadata.nrm["files"]:
            if tag in self.metadata.nrm["files"][file_name]:
                # Overwrite not allowed
                raise ValueError(f"'{file_name}' already has tag '{tag}'")
        else:
            if tag in self.metadata.nrm["classes"]:
                if data is not None:
                    save_method(
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
        if bulk == False:
            self.metadata.save(opj(self.path, "nrcm.json"))

    def processed_write(
        self,
        dpath=None,
        dtag_combinations=None,
        file_paths=None,
        alt_names=None,
        class_labels=None,
        process_chain=None,
    ):
        """
        Writes processed data to the dataset sourced from another NRCDataset or
        explicit file paths and file labels.

        Parameters
        ----------
        dpath : str, optional
            Path to the NRCDataset to be processed. If None, file_paths and
            class_labels must be specified. Default is None.
        dtag_combinations : list of list of str, optional
            List of list of tags to be combined to fetch files. If
            None, all classes are considered as separate tags. Default is None.
        file_paths : list of str, optional
            List of paths to the files to be processed. If None, dpath must be
            specified. Default is None.
        alt_names : list of str, optional
            List of alternative names for the files. If None, the original
            names are used. Default is None.
        class_labels : list of str, optional
            List of class labels for the files. If None, dpath must be
            specified. Default is None.
        process_chain : list of Processes, optional
            List of Process objects to be applied to the dataset. If None, no
            processing is done. Default is None.
        """

        if process_chain is None:
            process_chain = [self._default_process]
        if dpath is not None:
            base_dir, dataset_name = os.path.split(dpath)
            nrcd = NRCDataset(base_dir, dataset_name)
            extn = nrcd.metadata.nrm["file_ext"]

            if dtag_combinations is None:
                dtag_combinations = [[c] for c in nrcd.metadata.nrm["classes"]]
            fns, fls = nrcd.metadata.fetch(dtag_combinations)
            fps = [opj(nrcd.path, c, f"{fn}.{extn}") for c, fn in zip(fls, fns)]
            if alt_names is not None:
                assert len(alt_names) == len(fns)
                fns = alt_names
            fml = list(zip(fns, fls))

            print(f"NR > Processing {nrcd.dataset_name} to {self.dataset_name}")
            pb = tqdm(total=len(fns), desc="source processed")
            cp = self._chain_processes(process_chain, fps, fml)
            for data, label in cp:
                self.write(label[0], label[1], data, bulk=True)
                pb.update(1)
            pb.close()

        else:
            fns = [os.path.basename(fp).split(".")[0] for fp in file_paths]
            if alt_names is not None:
                assert len(alt_names) == len(fns)
                fns = alt_names
            fml = list(zip(fns, class_labels))

            print(f"NR > Processing files to {self.dataset_name}")
            pb = tqdm(total=len(fns), desc="source processed")
            cp = self._chain_processes(process_chain, file_paths, fml)
            for data, label in cp:
                self.write(label[0], label[1], data, bulk=True)
                pb.update(1)
            pb.close()
        self.metadata.save(opj(self.path, "nrcm.json"))

    def erase(self, file_name, tag=None, total_erase=False, bulk=False):
        """
        Erases data from the dataset

        Parameters
        ----------
        file_name : str
            Name of the file to be erased
        tag : str, optional
            Tag of the file to be erased. If None and total_erase is True, the
            file is completely removed from the dataset.
        total_erase : bool, optional
            Whether to completely remove the file from the dataset. Default is
            False.
        bulk : bool, optional
            Whether to save the metadata after erasing. Default is False.
        """

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
        if bulk == False:
            self.metadata.save(opj(self.path, "nrcm.json"))

    def register(self, file_type, ext="npy", data_shape=None):
        """
        Registers the dataset. This method should be called on a new dataset.

        Parameters
        ----------
        file_type : str
            Type of the files in the dataset
        ext : str, optional
            Extension of the files in the dataset. Default is 'npy'.
        data_shape : tuple of int, optional
            Shape of the data in the dataset. If None, the shape is inferred
            from the first file. Default is None.
        """
        dsr = True if data_shape is not None else False
        if self._de == True:
            if self._me == False:
                ld = os.listdir(self.path)
                classes = [i for i in ld if os.path.isdir(opj(self.path, i))]
                self.metadata = utils.NRCM(
                    dataset_name=self.dataset_name,
                    classes=classes,
                    file_type=file_type,
                    file_ext=ext,
                    data_shape=data_shape,
                )
                for fp in tqdm(glob.glob(opj(self.path, "**", f"*.{ext}"))):
                    if dsr == False:
                        d = np.load(fp)
                        self.metadata.nrm["data_shape"] = d.shape
                        dsr = True
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

    def _chain_processes(self, process_chain, paths, labels):
        cp = process_chain[0](paths=paths, labels=labels)
        for process in process_chain[1:]:
            cp = process(cp)
        return cp

    def _chain_process_transforms(self, process_chain):
        def cpt(path, label):
            data, label = process_chain[0].load(path), label
            # Even root process transform has to be called
            for process in process_chain:
                data, label = process.transform(data, label)
            return data, label

        return cpt

    def connect(
        self,
        tag_combinations=None,
        batch_size=None,
        shuffle=True,
        process_chain=None,
        categorical=True,
        enforce_binary=False,
        method="from_generator",
    ):
        """
        Connects to the dataset and returns a tf.data.Dataset object

        Parameters
        ----------
        tag_combinations : list of list of str, optional
            List of list of tags to be combined to form a single class. If
            None, all classes are considered as separate tags. Default is
            None.
        batch_size : int, optional
            Batch size of the dataset. If None, no batching is done. Default
            is None.
        shuffle : bool, optional
            Whether to shuffle the dataset. Default is True.
        process_chain : list of Processes, optional
            List of Process objects to be applied to the dataset. If None, no
            processing is done. Default is None.
        categorical : bool, optional
            Whether to convert the labels to one-hot encoded vectors. Default
            is True.
        enforce_binary : bool, optional
            Whether to enforce binary labels. If True, the labels are
            converted to 0 and 1. If False, the labels are converted to
            one-hot encoded vectors. Default is False.
        method : str, optional
            Method to be used for creating the dataset. If 'from_generator',
            the dataset is created using tf.data.Dataset.from_generator().
            If 'map_transforms', the dataset is created using
            tf.data.Dataset.map(). Default is 'from_generator'.


        Returns
        -------
        D : tf.data.Dataset
            Dataset object
        """

        if self._de == False:
            raise ValueError("connect(): called on non-existent dataset")
        if self._me == False:
            raise ValueError("connect(): called without nrcm.json")
        if tag_combinations is None:
            tag_combinations = [[cln] for cln in self.metadata.nrm["classes"]]

        extn = self.metadata.nrm["file_ext"]
        fns, fls = self.metadata.fetch(tag_combinations)
        uni = np.unique(fls)
        fps = [opj(self.path, cl, fn + f".{extn}") for cl, fn in zip(fls, fns)]
        ils = [np.where(uni == l)[0] for l in fls]
        n_classes = len(uni)
        if n_classes == 1:
            raise ValueError("Cannot connect to only a single class")
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
        if process_chain is None:
            process_chain = [self._default_process]

        if method == "from_generator":
            cprocess = self._chain_processes(process_chain, fps, ils)
            D = tf.data.Dataset.from_generator(
                lambda: cprocess, (tf.float32, tf.float32)
            )
        elif method == "map_transforms":
            cpt = self._chain_process_transforms(process_chain)
            D = tf.data.Dataset.from_tensor_slices((fps, ils))
            D = D.map(
                lambda path, label: tf.py_function(
                    cpt, inp=[path, label], Tout=[tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            # To filter out Exceptions in absence of the escaping yield trick
            # of Processes. This functionality is not removed from Processes
            # because they are used in writing too, where tf.data.Dataset is
            # not prepared
            D = D.ignore_errors()

        if shuffle:
            D.shuffle(len(fps))
        if batch_size is not None:
            D = D.batch(batch_size=batch_size)
        D = D.prefetch(tf.data.AUTOTUNE)
        print(f"NR > Connected to dataset {self.dataset_name}")
        return D

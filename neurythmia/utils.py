# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-29

# --Needed functionalities
# 1. AutoDecorator meta class

# --General
# 1. Starting Python 3.6 onwards, the standard dict maintains insertion order
#    by default, moreover, fns is sorted before extractinng labels therefore
#    the fetch() method of NRCM class is guaranteed to return the file names
#    in the same order everytime.

# ---DEPENDENCIES---------------------------------------------------------------
from typing import Iterable
import os

import json
import numpy as np

import matplotlib.pyplot as plt


# ---NRM------------------------------------------------------------------------
class NRCM:
    """
    Class to handle the metadata of a dataset. The metadata is stored in a json
    file with the following structure:
    {
        "dataset_name": <str>,
        "classes": <list[str]>,
        "file_type": <str>,
        "size": {
            <class_name>: <int>,
            ...
        },
        "data_shape": <tuple[int]>,
        "files": {
            <file_name>: [<tag>, ...],
            ...
        }
    }

    Parameters
    ----------
    file_type : str
        File type of the files in the dataset
    dataset_name : str
        Name of the dataset
    classes : list[str]
        List of classes in the dataset
    path : str
        Path to the json file containing the
    data_shape : tuple[int]
        Shape of the data in the files

    Returns
    -------
    nrm : NRCM
        Instance of the NRCM class


    Methods
    -------
    load(path)
        Loads the metadata from the json file at the given path
    save(path)
        Saves the metadata to the json file at the given path
    add(name, tag)
        Adds a file with the given name and tag to the metadata
    remove(name, tag=None, total_remove=False)
        Removes a file with the given name and tag from the metadata
    fetch_file(tags)
        Fetches the names of the files with the given tags
    update_size()
        Updates the size of the dataset
    info()
        Prints the metadata
    """

    def __init__(
        self,
        file_type=None,
        file_ext=None,
        dataset_name=None,
        classes=None,
        data_shape=None,
        path=None,
    ):
        self.path = path
        if self.path is not None:
            self.nrm = self.load(self.path)
        else:
            self.nrm = {}
            self.nrm["dataset_name"] = dataset_name
            self.nrm["classes"] = classes
            self.nrm["file_type"] = file_type
            self.nrm["size"] = {f"{cl}": 0 for cl in classes}
            self.nrm["data_shape"] = data_shape
            self.nrm["files"] = {}
            self.nrm["file_ext"] = file_ext

    def load(self, path):
        with open(path, "r") as f:
            nrm = json.load(f)
            return nrm

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.nrm, f, indent=4)

    def add(self, name, tag):
        if name not in self.nrm["files"]:
            self.nrm["files"][name] = [tag]
        else:
            if tag in self.nrm["files"][name]:
                raise ValueError(f"File '{name}' already has tag <{tag}>")
            else:
                self.nrm["files"][name].append(tag)

    def remove(self, name, tag=None, total_remove=False):
        if name not in self.nrm["files"]:
            raise ValueError(f"File '{name}' does not exist")
        else:
            if total_remove:
                del self.nrm["files"][name]
            else:
                if tag not in self.nrm["files"][name]:
                    raise ValueError(f"File '{name}' does not have tag <{tag}>")
                self.nrm["files"][name].remove(tag)

    def fetch(self, tag_combinations: Iterable[Iterable[str]]):
        file_names = []
        file_label = []
        for tc in tag_combinations:
            for file_name in self.nrm["files"]:
                if all([tag in self.nrm["files"][file_name] for tag in tc]):
                    file_names.append(file_name)
                    file_label.append(self.nrm["files"][file_name][0])

        fns = sorted(list(set(file_names)))  # sorted for reproducible order
        fls = [file_label[file_names.index(fn)] for fn in fns]
        return fns, fls

    def info(self):
        print(f"NR {self.nrm['dataset_name']} > NRCM Info:")
        for key in self.nrm:
            if key != "files":
                print(f"{key}: {self.nrm[key]}")


# ---EXTERNAL METADATA UTILS----------------------------------------------------
def extract_timestamps(time_stamp: str):
    """
    Extracts the timestamps from the given string

    Parameters
    ----------
    time_stamp : str
        String containing the timestamps

    Returns
    -------
    t1s : int
        Start time in seconds
    t2s : int
        End time in seconds
    duration : int
        Duration in seconds
    """
    di = {"hr": 3600, "min": 60, "sec": 1}
    co = time_stamp.split("to")
    t1 = co[0].split()
    t2 = co[1].split() if len(co) > 1 else None
    t1s = 0
    t2s = 0

    for i in range(0, len(t1), 2):
        t1s += int(t1[i]) * di[t1[i + 1]]
    if type(t2) != type(None):
        for i in range(0, len(t2), 2):
            t2s += int(t2[i]) * di[t2[i + 1]]

    duration = abs(t2s - t1s)

    return t1s, t2s, duration


# ---VISUALIZATION UTILS--------------------------------------------------------
def plot_tfd(tfd, sf, title="test", cmap="magma", figsize=(10, 5)):
    """
    Plots the time-frequency distribution

    Parameters
    ----------
    tfd : np.ndarray
        Time-frequency distribution
    sf : int
        Sampling frequency
    title : str, optional
        Title of the plot, by default "test"
    cmap : str, optional
        Colormap to use, by default "magma"
    figsize : tuple, optional
        Figure size, by default (10, 5)
    """
    freq_axis = np.arange(tfd.shape[0]) / sf
    time_axis = np.arange(tfd.shape[1]) / sf

    plt.figure(figsize=figsize)
    plt.pcolormesh(time_axis, freq_axis, tfd, cmap=cmap)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()

    plt.show()


# ---AUTO DECORATOR-------------------------------------------------------------

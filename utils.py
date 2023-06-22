# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-20

# --Needed functionalities


# ---DEPENDENCIES---------------------------------------------------------------
from typing import Iterable

import json


# ---NRM------------------------------------------------------------------------
class NRMDataset:
    def __init__(self, file_extension=None, dataset_name=None, classes=None, path=None):
        self.path = path
        if self.path is not None:
            self.nrm = self.load(self.path)
        else:
            self.nrm = {}
            self.nrm["dataset_name"] = dataset_name
            self.nrm["classes"] = classes
            self.nrm["file_extension"] = file_extension
            self.nrm["size"] = {f"{cl}": 0 for cl in classes}
            self.nrm["data_shape"] = None
            self.nrm["files"] = {}

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

    def fetch_file(self, tags: Iterable[str]):
        file_names = []
        for file_name in self.nrm["files"]:
            if all([tag in self.nrm["files"][file_name] for tag in tags]):
                file_names.append(file_name)
        return file_names

    def update_size(self):
        for cl in self.nrm["size"]:
            self.nrm["size"][cl] = len(self.fetch([cl]))

    def info(self):
        print("NR >> NRMD Info: ")
        for key in self.nrm and key != "files":
            print(f"{key}: {self.nrm[key]}")

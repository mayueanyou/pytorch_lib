import os,sys

class CLDataset():
    def __init__(self,yaml_path) -> None:
        self.name = type(self).__name__
        #self.yaml_
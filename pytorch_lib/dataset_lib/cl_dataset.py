import os,sys

class CLDataset():
    def __init__(self,train_dataset,test_dataset) -> None:
        self.name = type(self).__name__
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    
    #def get_loader:
    #    ...
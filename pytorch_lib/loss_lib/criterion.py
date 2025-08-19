import os,sys,torch
from abc import ABC,abstractmethod




class Criterion(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @abstractmethod
    def calculate_performance(self):pass
    
    @abstractmethod
    def calculate_loss(self):pass
    
    def is_better(self,current_performance,current_loss,best_performance,best_loss):
        if current_performance > best_performance: return True
        elif current_performance == best_performance and current_loss < best_loss: return True
        else: return False
    
    def print_result(self,val_performance,val_loss,test_performance,test_loss,
                     best_val_performance,best_val_loss,best_test_performance,best_test_loss):
        print(f"Validate: \n Performance: {val_performance:>0.4f}, Avg loss: {val_loss:>8f} \n")
        print(f"Test: \n Performance: {test_performance:>0.4f}, Avg loss: {test_loss:>8f} \n")
        print(f"Best Validate: \n Performance: {best_val_performance:>0.4f}, Avg loss: {best_val_loss:>8f} \n")
        print(f"Best Test: \n Performance: {best_test_performance:>0.4f}, Avg loss: {best_test_loss:>8f} \n")
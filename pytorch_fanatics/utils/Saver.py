import torch
import numpy as np

class Saver:
    def __init__(self,path="./",mode="max"):
        self.path=path
        self.mode=mode
        self.best_score=None
        if self.mode == "min":
            self.score = np.Inf
        else:
            self.score = -np.Inf
    def save(self,model,optimizer,scheduler,metric=None):
        if metric is not None:
            if self.mode == "min":
                score = -1.0 * metric
            else:
                score = np.copy(metric)
            if self.best_score is None:
                self.best_score = score
                self.saved(metric,model,optimizer,scheduler,"best.pth")
            else:
                self.best_score=score
                self.saved(metric,model,optimizer,scheduler,"best.pth")
    def saved(self,metric,model,optimizer,scheduler,char):
        check={"model_state_dict"     : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "scheduler_state_dict" : scheduler.state_dict()                                    
                }
        if metric not in [-np.inf, np.inf, -np.nan, np.nan]:
            torch.save(check,self.path+char)
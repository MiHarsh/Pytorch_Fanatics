import torch

class Logger:
    def __init__(self,path="./"):
        self.path=path
    def write(self,message,verbose=1):
        if verbose==1:
            print(message)
        with open(self.path+"log.txt","a+") as f:
            f.write(message+"\n")

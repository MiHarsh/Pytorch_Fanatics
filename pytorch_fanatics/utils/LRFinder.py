import numpy as np
import torch
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

class LRFinder:
    def __init__(self,model,train_dataloader,optimizer,device,initial_lr=1e-8,final_lr=10,beta=0.98):
        self.model=model
        self.best_weights=model.state_dict()
        self.tl=train_dataloader
        self.optimizer=optimizer
        self.beta=beta
        self.i=initial_lr
        self.f=final_lr
        self.device=device
        self.optimizer.param_groups[0]['lr'] = self.initial_lr
        self.num = len(self.tl)-1
        self.factor = (self.f-self.i)/(2*self.num)
        self.loss_list=[]
        self.log_lrs=[]
    def find(self):
        best_loss,avg_loss,smoothed_loss,out_flag=np.inf,0.0,0.0,0
        self.model.train()
        for epoch in tqdm(arange(3),total=3,position=0,leave=True): # ==> We can find LR in 2 epochs
            tk = tqdm(self.tl, total=len(self.tl), position=0, leave=True)
            for i,data in enumerate(tk):
                self.optimizer.zero_grad()
                for key, value in data.items():
                    data[key] = value.to(self.device)
                _, loss = self.model(**data)
                loss.backward()
                self.optimizer.step()
                avg_loss = self.beta * avg_loss + (1-self.beta) *loss.data.item()
                #Compute the smoothed loss
                smoothed_loss += avg_loss / (1 - self.beta**(i+1))   
                #Stop if the loss is exploding
                if i > 0 and smoothed_loss > 1000 * best_loss:
                    out_flag=1
                    break
                #Record the best loss
                if smoothed_loss < best_loss or i==0:
                    best_loss = smoothed_loss
                    self.best_weights=  self.model.state_dict()
                    
                #Store the values
                self.loss_list.append(smoothed_loss/len(self.tl))
                self.log_lrs.append(np.log10(self.i))
                #Update the lr for the next step
                self.i += self.factor
                self.optimizer.param_groups[0]['lr'] = self.i
            if out_flag==1:
                print("LR Finder is Complete, use LRFinder.plot() to view the plot.")
                break
        return self.model.load_state_dict(self.best_weights)
        
                
    def plot(self):
        plt.plot( self.log_lrs, self.loss_list,color='blue')
        plt.title('Plot b/w losses and lrs')
        plt.xlabel('LRs----> 1e')
        plt.ylabel('Loss----->')
        plt.show()


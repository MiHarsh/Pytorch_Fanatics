import numpy as np
import pandas as pd
import torch
import sys
from tqdm.notebook import tqdm

try:
    from torch.cuda import amp
    _amp_available = True
except ImportError:
    _amp_available = False

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        train_scheduler = None,
        val_scheduler  = None,
        accumulation_steps=1,
        fp16=False,
        use_mean_loss=False,
        checkpoint=None,
        save_path = "./"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_scheduler = train_scheduler
        self.val_scheduler=val_scheduler
        self.accumulation_steps = accumulation_steps
        self.fp16 = fp16
        self.use_mean_loss=use_mean_loss
        self.last_idx = 0
        self.checkpoint=checkpoint
        self.save_path = save_path
        if checkpoint is not None:
            print("Loading Checkpoint...Please wait")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.train_scheduler is not None:
                self.train_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.val_scheduler is not None:
                self.val_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.last_idx = checkpoint['last_idx']
        if self.fp16 and not _amp_available:
            raise Exception(
                "You want to use fp16 but dont have amp installed"
            )
        self.scaler = None
        if self.fp16:
            self.scaler = amp.GradScaler()
        

    def train(self, data_loader):
        self.model.train()
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()
        self.train_loss=0.0 
        if self.checkpoint is not None:
            self.train_loss = self.checkpoint["train_loss"]
        tk = tqdm(data_loader, total=len(data_loader), position=0 , initial = self.last_idx,leave=True)
        for idx, data in enumerate(tk):
            if self.accumulation_steps == 1 and idx == 0:
                self.optimizer.zero_grad()
            for key, value in data.items():
                data[key] = value.to(self.device)
            if self.fp16:
                with amp.autocast():
                    outputs, loss = self.model(**data)
                if self.use_mean_loss:
                    loss=loss.mean()
                self.scaler.scale(loss).backward()
            else:
                outputs, loss = self.model(**data)
                if self.use_mean_loss:
                    loss=loss.mean()
                loss.backward()
            if (idx + 1) % self.accumulation_steps == 0:
                if self.fp16:
                    self.scaler.step(self.optimizer)
                else:
                    self.optimizer.step()
                if self.train_scheduler is not None:
                    self.train_scheduler.step()
                if self.fp16:
                    self.scaler.update()
                if idx > 0:
                    self.optimizer.zero_grad()            
            self.train_loss += (loss.data.item())/len(data_loader)
            self.last_idx = idx
            tk.set_description(f"current_loss: {loss.data.item():.4f} loss(avg): {self.train_loss:.4f}")
            
        return self.train_loss

    def evaluate(self, data_loader):
        self.model.eval()
        outs    = []
        targets = []
        val_loss = 0.0
        self.last_idx = 0
        with torch.no_grad():
            tk = tqdm(data_loader, total=len(data_loader), position=0,leave=True)
            for idx, data in enumerate(tk):
                for key, value in data.items():
                    data[key] = value.to(self.device)
                if self.fp16:
                    with amp.autocast():
                        outputs, loss = self.model(**data)
                    if self.use_mean_loss:
                        loss=loss.mean()
                else:
                    outputs, loss = self.model(**data)
                    if self.use_mean_loss:
                        loss=loss.mean()
                val_loss += (loss.data.item())/len(data_loader)
                outs.extend(outputs.cpu().numpy())
                targets.extend(data['targets'].cpu().numpy())
                tk.set_description(f"current_loss: {loss.data.item():.4f} loss(avg): {val_loss:.4f}")
        if self.val_scheduler is not None:
            if isinstance(self.val_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.val_scheduler.step(val_loss)
            else:
                self.val_scheduler.step()
        return np.array(targets), np.array(outs),val_loss

    def predict(self, data_loader):
        self.model.eval()
        final_predictions = []
        with torch.no_grad():
            tk = tqdm(data_loader, total=len(data_loader), position=0,leave=True)
            for idx, data in enumerate(tk):
                for key, value in data.items():
                    data[key] = value.to(self.device)
                outputs, loss = self.model(**data)
                final_predictions.extend(outputs.cpu().numpy())
            tk.close()
        return np.array(final_predictions)

    def saver(self):
        check={"model_state_dict"     : self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "last_idx"            : self.last_idx ,
                "train_loss"          : self.train_loss,                     
                }
        if self.train_scheduler is not None:
            check["scheduler_state_dict"] = self.train_scheduler.state_dict()
        elif self.val_scheduler is not None:
            check["scheduler_state_dict"] = self.val_scheduler.state_dict()
        else:
            check["scheduler_state_dict"] = None
        torch.save(check,self.save_path+"last_checkpoint.pth")
        
        


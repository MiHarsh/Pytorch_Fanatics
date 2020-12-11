# Pytorch_Fanatics

Pytorch_Fanatics is a Python library for ease of Computer Vision tasks.This contains a bunch of various tools which will help to create customized codes to train CV models.

##### This library includes:
```bash
* Dataset class
* LRFinder
* EarlyStopping
* Trainer
* Logger
* Saver
```
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pytorch_fanatics.

```bash
pip install pytorch_fanatics
```

## Usage
#### 1) Dataset Class
```python
from pytorch_fanatics.dataloader import Cloader

dataset    = Cloader(image_path,targets,resize=None,transforms=None)
""" returns {'image':tensor_image,'targets':tensor_labels} """
dataloader = DataLoader(dataset,batch_size=64
                                     ,shuffle=True,num_workers=4)
                                     
""" using next(iter(dataloader)) returns a dictionary with keys 'image'
 and 'targets'."""
```

#### 2) LR Finder
```python
from pytorch_fanatics.utils import LRFinder

lr_finder = LRFinder(model,train_dataloader,optimizer,device
                            ,initial_lr=1e-8,final_lr=10,beta=0.98) 
""" This creates an object, finds the lr based on the optimizer used."""

lr_finder.find()       """ To find the lr """
lr_finder.plot()       """ Plots the graph (Loss V/S lr) """

""" LRFinder starts the LR from inital_lr(which is kept small) and 
    gradually increases the lr. Please finetune your model for 2-3 epochs 
    and then use this for better results .""" 
```

#### 2) Early Stopping
```python
from pytorch_fanatics.utils import EarlyStop

es = EarlyStop(patience=7, mode="max", delta=0.0001) 

""" Sample code """
for epoch in range(epochs):
    epoch_score = Trainer.evaluate(......)
    es(epoch_score , model , model_path ="./best.pth")
    """ model_path is the location+filename to save the best model """
    if es.early_stop=True:
	    break

es.reset()  """ resets the best_epoch_score, if in case training multiple
                folds without creating 'es' object again and again."""
```

#### 4) Trainer
```python
from pytorch_fanatics.trainer import Trainer

trainer        = Trainer(model,optimizer,device,train_scheduler = None,
                 val_scheduler = None,accumulation_steps=1,fp16=False,
                use_mean_loss=False,checkpoint=None,save_path = "./")
                
""" Training and Evaluating """
train_loss              = trainer.train(train_dataloader)
y_true ,y_pred,val_loss = trainer.evaluate(val_dataloader)

""" Prediction """
y_preds                 = trainer.predict(test_dataloader)  

""" In depths """
""" train_scheduler/val_scheduler : call scheduler.step() while training
                                    or after validating
    accumulation_step             : implements gradient accumulation 
                                    (default = 1)
    fp16                          : mixed precision training
    use_mean_loss                 : loss.mean().backward()
                                    (false if loss is already meaned)
    checkpoint                    : torch.load(checkpoint),loads the 
                                    checkpoint and resumes training.
    save_path                     : location to save the last epoch 
                                    weights                             """

""" Having problem with non resumable training epochs? """    
trainer.saver()  """ saves the last epoch to the save_path location """

""" dataloader must be set in the same state to resume training 
Use [https://gist.github.com/usamec/1b3b4dcbafad2d58faa71a9633eea6a5]
an implementation of ResumableRandomSampler()  """

sampler = ResumableRandomSampler(dataset)
loader  = DataLoader(dataset, batch_size=64, sampler=sampler)
torch.save(sampler.get_state(), "test_samp.pth") -- save the state
sampler.set_state(torch.load("test_samp.pth"))  -- load and set the state

""" and tadaa , resume training 100 % :) """
```

#### 5) Logger
```python
from pytorch_fanatics.logger import Logger

logger   = Logger(path="./")  """ save path for logger"""
logger.write( message ,verbose = 1) """ verbose to print the message"""

""" Helps Keep Track of Training """
```

#### 6) Saver
```python
from pytorch_fanatics.utils import Saver

saver    = Saver(path="./" , mode = "max")
""" saves the model, optimizer and scheduler based on score and mode """

saver.save(model,optimizer,scheduler,metric)
```

---
**NOTE ( Regarding model )**

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base_model = timm.create_model('resnet18',pretrained=True,num_classes=1)
    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        out = self.base_model(image)
        loss = nn.BCEWithLogitsLoss()(out.view(batch_size,), targets.type_as(out))
        return out, loss
```
---

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://github.com/MiHarsh/Pytorch_Fanatics/blob/main/LICENSE)

## References
* FastAi Documentations for LRFinder.
* Pytorch Documentations.
* Abhishek Thakur Sir's Github Project wtfml(Awesome work Have a look at it too).

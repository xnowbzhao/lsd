import os
from tqdm import tqdm
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
import numpy


class Trainer():
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''



    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, label):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()

        loss= self.compute_loss(data, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        val=0.0
        num=0
        for i in range(val_loader.length()):
            tdata, tlabel=val_loader.generate_batch(i)
            for j in range(tdata.shape[0]):
                loss = self.eval_step(tdata[j], tlabel[j])
                val=val+torch.sum(loss.float())
                num=num+tdata[j].shape[0]

        return val/num
    
    def eval_step(self, data, label):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        
        data =torch.tensor(data).to(self.device).float()
        label = torch.tensor(label).to(self.device).float()        

        with torch.no_grad():
            loss = self.model.compute_loss(data, label)
        return  loss

    def compute_loss(self, data, label):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        data =torch.tensor(data).to(self.device).float()
        label = torch.tensor(label).to(self.device).float()   
        output = self.model.pred(data)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, label)

        return loss.float()



    

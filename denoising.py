import torch
import torch.optim as optim
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from tensorboardX import SummaryWriter
import argparse
import time
import config
from trainer import Trainer
from checkpoints import CheckpointIO
from fileloader import Loader
import pickle
import sys
if __name__ == '__main__':  
# Arguments

    is_cuda = (torch.cuda.is_available() )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    listfile=open(sys.argv[2],"r")
    
    
    model = config.get_model(device)
    model.eval()
    checkpoint_io = CheckpointIO(sys.argv[1], model=model)
    try:
        load_dict = checkpoint_io.load('model_best.pt')
    except FileExistsError:
        print("error")
        input()
    
    filenumber= int(listfile.readline())
    output=[]

    for i in range(filenumber):
        filename=listfile.readline().strip('\n')
        data=np.fromfile(filename,dtype="float32").reshape((-1,80,80,3))

        d_split = np.array_split(data, 10, axis=0)
        for j in range(len(d_split)):
            with torch.no_grad():
                c = model.pred(torch.from_numpy(d_split[j]).to(device))
            temp=c.detach().cpu().numpy()

            output.append(temp)
    
    output=np.concatenate(output,axis=0)

    output=output.reshape(-1)

    output.tofile("normal.bin")
    listfile.close()

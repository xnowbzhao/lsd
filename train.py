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

if __name__ == '__main__':  
# Arguments

    is_cuda = (torch.cuda.is_available() )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    np.random.seed(0)
    # Set t0
    t0 = time.time()

    model = config.get_model(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, device=device)
    # Shorthands
    out_dir = 'out/'
    logfile = open('out/log.txt','w')
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = np.inf

    logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    batch_size=80
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    nparameters = sum(p.numel() for p in model.parameters())

    logfile.write('Total number of parameters: %d\n' % nparameters)

    #test_loader=Loader("test/",batch_size)
    dev_loader=Loader("dev/",batch_size)
    train_loader=Loader("train/",batch_size)

    for epoch_it in range(0,20):
        logfile.flush()
        
        for i in range(train_loader.length()):
            tdata, tlabel=train_loader.generate_batch(i)
            for j in range(tdata.shape[0]):
                loss = trainer.train_step(tdata[j], tlabel[j])
                logger.add_scalar('train/loss', loss, it)
            print('[Epoch %02d] file=%02d: loss=%.6f' % (epoch_it, i, loss))
            logfile.write('[Epoch %02d] file=%02d: loss=%.6f\n' % (epoch_it, i, loss))
             


        logfile.write('Saving checkpoint')
        checkpoint_io.save('model.pt', epoch_it=epoch_it, loss_val_best=metric_val_best)

        #metric_val = trainer.evaluate(test_loader)
        #metric_val=metric_val.float()
        #logfile.write('Test metric : %.6f\n'  % (metric_val))
        
        metric_val2 = trainer.evaluate(dev_loader)
        metric_val2=metric_val2.float()
        logfile.write('Validation metric : %.6f\n' % (metric_val2))
        
        if metric_val2 < metric_val_best:
            metric_val_best = metric_val2
            logfile.write('New best model (loss %.6f)\n' % metric_val_best)
            checkpoint_io.save('model_best.pt', epoch_it=epoch_it, loss_val_best=metric_val_best)
        checkpoint_io.save('model_'+str(epoch_it)+'.pt', epoch_it=epoch_it, loss_val_best=metric_val_best)
    logger.close()
    
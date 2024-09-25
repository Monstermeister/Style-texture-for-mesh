import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler

import numpy as np
import os
import random
from tqdm import tqdm
from model.PVCNN2 import PVCNN2
from train.utils import MSE,L1_loss,KL_loss, criterion

class shape_train:
    """
    Training System



    """

    def __init__(self,
                 Dataset,
                 batch_size,
                 seeds,
                 epochs,
                 device='cuda',
                 deterministic=True,
                 path= '/home/jiun/work/Brian_Lab/UVDiff-Designer/path'
                 ):
        # path
        self.path         = path   ## root path for saving
        self.save_path    = os.path.join(path, 'result') 
        os.makedirs(self.save_path, exist_ok=True)
        # CUDA Device
        self.device       = device
        if self.device == 'cuda':
          cudnn.benchmark = True
          if deterministic == True:
            cudnn.deterministic = True
            cudnn.benchmark     = False
        
        # Random Seed
        if seeds is None:
            self.seed = torch.initial_seed() % (2**32 -1)
       
        self.seed          = seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


        # Dataset setup -> should split system train, evaluation check option
        self.dataset      = Dataset     
        self.batch_size   = batch_size
        self.max_epoch    = epochs    
        self.start_epoch  = 0


    def train(self):
        self.init_train()
       
        try:
            self.train_in_epoch()
        
        except Exception:
            raise
        
        finally:
            self.after_train()


    def init_train(self):
        ##### Data loader
        self.Dataloader     = DataLoader(self.dataset,
                                         batch_size=8,
                                         pin_memory=True,
                                         sampler =RandomSampler(self.dataset, num_samples = len(self.dataset)*8,
                                         generator=torch.Generator().manual_seed(self.seed)))
        self.eval_loader    = DataLoader(self.dataset, batch_size=1, pin_memory=True)

        self.max_iter       = len(self.Dataloader)
        ##### model
        self.model          = PVCNN2(num_classes=2,
                                     extra_feature_channels=3,
                                     width_multiplier=1,  #0.5
                                     voxel_resolution_multiplier=1,
                                     ).to(self.device)   
       
        self.loss           =   criterion         
        self.optimizer      =   torch.optim.Adam(self.model.parameters(),
                                                 lr = 1e-3,
                                                 weight_decay = 1e-5
                                                 )

        
        self.lr_scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer,
                                                                  last_epoch = self.start_epoch-1,
                                                                  T_max =self.max_epoch,
                                                                  )

      # every epoch evaluation and record the results using these path.  
        self.latest_ckpt    =  os.path.join(self.save_path,'latest.pth')
        self.best_ckpt      =  os.path.join(self.save_path,'best.pth')
       


    def train_in_epoch(self):
        
        ##### epoch
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
        


    def after_train(self):
        ####5.visualize log information(should be listed up)
        pass    

    def before_epoch(self):
        ### add usage
        self.eval_interval = 1
        
    def train_in_iter(self):
        self.iter = 0
        #### call dataloader
        for (inp, tar)  in tqdm(self.Dataloader,total = len(self.Dataloader), desc = 'train', ncols = 0, ascii = ' =', leave = True,):
           
           self.iter += 1
           print(f'\niteration:{self.iter}/{self.max_iter}')
           self.before_iter()
           self.train_one_iter(inp,tar)        
           self.after_iter()

    
    def after_epoch(self):
        self.model.eval()
        with torch.no_grad():
            loss =0
            for (inp,tar) in tqdm(self.eval_loader,total = len(self.eval_loader),  desc ='eval', ncols = 0, ascii = ' =', leave = True,):
                inp      = inp.to(self.device)
                tar      = tar.to(self.device)

                out ,_,_ = self.model(inp)
                loss     += MSE(tar,out)          #find more metrics
        
        print(f'Epochs:{self.epoch},MSE:{loss}')
        torch.save({
                    'epoch':self.epoch,
                    'model':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'loss': loss,
                    },self.latest_ckpt)
       

         

    def before_iter(self):
        ###   print current iteration
        print(f'total_iteration:{self.iter+ self.epoch*self.max_iter}')
        # opt training mode
        self.model.train()

    def train_one_iter(self,inp, tar):
        inp        = inp.to(self.device)
        tar        = tar.to(self.device)

        self.optimizer.zero_grad()
        out,latent_mean, latent_var        = self.model(inp)
        print(out)
        loss                               = self.loss(out,latent_mean,latent_var, tar).to(self.device)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        
        
    def after_iter(self):
        if self.iter != self.max_iter:
            print("Next iteration")
        else:
            pass
        













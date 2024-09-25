import torch
import numpy as np
import torch.nn as nn




def MSE(tar,out):
    loss = nn.MSELoss(reduction='none')
    return torch.mean(torch.sum(loss(tar,out), dim=1))


def L1_loss(tar,out):
    loss = nn.L1Loss(reduction='none')
    return torch.mean(torch.sum(loss(tar,out),dim=1))


def KL_loss(mean,var):
    
    KLD    = -0.5* torch.sum((1+var -mean.pow(2) -var.exp()),dim=1)
    return torch.mean(KLD)

def criterion(out, mean, var,tar):
    
    loss  = MSE(tar,out) + KL_loss(mean,var)

    return loss
    




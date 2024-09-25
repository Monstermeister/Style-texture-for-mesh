

import os
import numpy as np
import torch
import PIL
from torch.utils.data import  DataLoader
from data.base import Shape_data,Content_data, Style_data,Content_img
from train.train import shape_train


shape_dataset = Shape_data('data/Deepfashion3D/filtered_registered_mesh',4096, with_normal=True)


if __name__ == "__main__":
   
    trainer = shape_train(shape_dataset,8,1588147245,5)
    trainer.train()





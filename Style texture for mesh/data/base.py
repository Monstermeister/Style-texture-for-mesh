"""
Base data process

Content data 
  read content dataset 
  return content dataset
Style data
  read style dataset
  return style dataset

"""



import os
import glob
import numpy as np
import PIL
from   PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import trimesh


def Content_img(con_path, target_size,content_index =0):
    resize      = transforms.Resize(target_size,interpolation= transforms.InterpolationMode.BICUBIC, antialias=True)
    fname_lists = os.listdir(con_path)
    data_file   = os.path.join(con_path, fname_lists[content_index])
    image_file  = glob.glob(f'{data_file}/*.png')[0]
    image       = resize(torch.tensor(np.array(PIL.Image.open(image_file)).astype(np.uint8)).permute(2,0,1)/255.)
    
    return image.reshape(1,-1,image.size(1),image.size(2))



class Shape_data(Dataset):
    
    """
    shape_path: shape path(mesh)
    """
    def __init__(self,
                 shape_path,
                 num_points,
                 with_normal,
                 ):
        self.shape_path  = shape_path
        self.fname_lists = os.listdir(self.shape_path)
        self.num_points  = num_points
        self.with_normal = with_normal
    def __len__(self):
        
        return len(self.fname_lists)


    def __getitem__(self, index):
    # For shape training we should sample point dataset into same size(point number), 
    # Output UV coordinate information, which should be same size.
       
        data_file             = os.path.join(self.shape_path, self.fname_lists[index])
        pointset, uv          = self.read_shape(data_file)
        

        return pointset, uv
       
        

    def read_shape(self, path):
        if len(glob.glob(f'{path}/*.obj'))==0:
            print(path)
        mesh_file     = glob.glob(f'{path}/*.obj')[0]
       
        mesh          = trimesh.load(mesh_file)
        # normalize points
        points        = self.normalize(mesh.vertices)
        # np random choice
        choice        = np.random.choice(points.shape[0],self.num_points)
        coords        = points[choice,:].transpose(0,1)
        uv            = torch.Tensor(mesh.visual.uv[choice,:]).transpose(0,1)
        # point with normal
        if self.with_normal:
            normal    = torch.Tensor(mesh.vertex_normals[choice,:]).transpose(0,1)
            pointset  = torch.cat([coords, normal])
        else:
            
            pointset  = coords

       # mtl_file      = glob.glob(f'{path}/*.obj.mtl')
        

        return pointset, uv


    def normalize(self,points):
        points     = torch.Tensor(points)
        centroid   = torch.mean(points, axis=0)
        points     = points - centroid
        
        return points / torch.max(torch.linalg.norm(points, axis=1))
    
    






class Content_data(Dataset):
    """
    con_path: content path(texture image)
    img_size: train image size
    split: train or valid
    
    """
    def __init__(self,
                 con_path,
                 target_size,
                 split = 'train',
                 ):

        
        self.con_path   = con_path
        self.split      = split
        self.fname_lists = os.listdir(self.con_path)
        self.transforms = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)  

    def __len__(self):
        """
        return dataset length: Content dataset length 
        """
        return len(self.fname_lists)               
   

                
    def __getitem__(self, index):
        
        """
        read content image(texture image), shape information
        
        resize image
        
           
      
        return content image, shape information. 
        """
        data_file = os.path.join(self.con_path, self.fname_lists[index]) 
        image      = self.resize(self.read_img(data_file))
        shape      = self.read_shape(data_file)

        return {'texture': image, 'shape': shape}

    
    def read_img(self, path):
        """
        read img from path
        [2048,2048,4] 4-channel: alpha channel image(masked)
        """
        image_file = glob.glob(f'{path}/*.png')[0]
        image      = PIL.Image.open(image_file)#.convert('RGB')
        image      = torch.tensor(np.array(image).astype(np.uint8))     #### if we use pixel unshuffle structure in the first layer(512,512,64), then we would not apply interpolation.
        
        return image.permute(2,0,1) 
   
    def resize(self, img):
        """
        Resize image
        """
        return self.transforms(img/255.)
    
    def read_shape(self,path):
        """
        read shape information
        """
        mesh_file     = glob.glob(f'{path}/*.obj')[0]
        mesh          = trimesh.load(mesh_file)
        
        mtl_file      = glob.glob(f'{path}/*.obj.mtl')
        return {'uv_coord': torch.tensor(mesh.visual.uv), 'face_normal': torch.tensor(mesh.face_normals), 'origin_mtl_path': mtl_file}
 
    @staticmethod
    def normalize(img):
        """
        [0,1] -> [-1,1]
        """
        return 2*img-1

    @staticmethod
    def denormalize(img):
        """
        [-1,1] -> [0,1]
        """
        return (img+1)/2





class Style_data(Dataset):
  
    def __init__(self,
                  style_path,
                  target_size,
                  style_name ='',
                  split ='train'
                  ):
        self.style_path   = style_path
        self.style_name   = style_name
        self.split        = split

        self.transforms   = transforms.Resize(target_size, interpolation = transforms.InterpolationMode.BICUBIC, antialias=True)

        self.data_file    = os.path.join(self.style_path,self.style_name)
        self.image_lists  = os.listdir(self.data_file)

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, index):

        image     = self.resize(self.read_img(os.path.join(self.data_file,self.image_lists[index])))
        return image

    def read_img(self, path):

        image     = PIL.Image.open(path)
        image     = torch.tensor(np.array(image).astype(np.uint8))

        return image.permute(2,0,1)

    def resize(self, img):

        return self.transforms(img/255.)

    @staticmethod
    def normalize(img):
        return 2*img-1

    @staticmethod
    def denormalize(img):
        return (img+1)/2

import torch.nn as nn
import torch
from model.utils import create_pointnet2_sa_components, create_pointnet2_fp_modules, create_mlp_components

from model.modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule

__all__ =['PVCNN2']


class PVCNN2(nn.Module):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256))), #### (None, (16,0,8, 32, (256,256,512))
        
    ]
    fp_blocks = [
        ((256, 256), (256, 1, 8)),
        ((256, 256), (256, 1, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 1, 32)),
    ]

    def __init__(self, num_classes, extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels  = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers    = nn.ModuleList(sa_layers)
        #LATENT CODE
        #self.z_mean -> sharedMLP
        #self.z_var  -> sharedMLP
        #self.reparameterization
        
        self.z_mean = SharedMLP(channels_sa_features,channels_sa_features,dim=1)
        self.z_var  = SharedMLP(channels_sa_features,channels_sa_features,dim=1)
        #self.act    = nn.ReLU(True)
        
        

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers    = nn.ModuleList(fp_layers)

        layers, _         = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier   = nn.Sequential(*layers)
        self.normalizer   = nn.Sigmoid()

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs       = inputs['features']

        coords, features = inputs[:, :3, :].contiguous(), inputs
        
        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
           
        in_features_list[0]  = inputs[:, 3:, :].contiguous()
        
        
        latent_mean          = self.z_mean(features)
        latent_var           = self.z_var(features)
       
        latent               = self.reparameterization(latent_mean, latent_var)
                
        
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            latent, coords = fp_blocks((coords_list[-1-fp_idx], coords,latent, in_features_list[-1-fp_idx]))
                   
        
        return self.normalizer(self.classifier(latent)), latent_mean, latent_var


    def reparameterization(self, mean, var):
        
        var      =  torch.exp(0.5 * var)
        epsilon  =  torch.randn_like(var).to('cuda')
        z        =  mean + var * epsilon

        return   z 










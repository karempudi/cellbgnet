import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.optimizer import Optimizer


class CoordConv(nn.Module):
    """
    CoordConv class, input goes through conv layers,
    and cell bg coordinates also go through conv layer and
    get added to the result
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CoordConv, self).__init__()
        self.conv2d_im = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, padding=padding)
        self.conv2d_coord = nn.Conv2d(in_channels=1, out_channels=out_channels,
                                      kernel_size=1, padding=0)
        
    def forward(self, input, cell_bg_coord):
        ret_1 = self.conv2d_im(input)
        ret_2 = self.conv2d_coord(cell_bg_coord)
        return ret_1 + ret_2

class UnetBGConv(nn.Module):
    """
    Used for both frame analysis and contex module.. In our
    case, context module is just an added U-net on top, so that
    the model becomes double U-nets, concatenated.    
    Cell Bg coordinate can be added on top if you set use_coordconv=True 
    """
    def __init__(self, n_inp, n_filters=64, n_stages=5, pad=1, ker_size=3,
                  use_coordconv=False, device='cuda:0'):
        super(UnetBGConv, self).__init__()
        curr_N = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()
        self.use_coordconv = use_coordconv
        self.device = device

        if self.use_coordconv:
            self.layer_path.append(
                CoordConv(in_channels=n_inp, out_channels=curr_N,
                          kernel_size=ker_size, padding=pad).to(device)
            )
        else:
            self.layer_path.append(
                nn.Conv2d(in_channels=n_inp, out_channels=curr_N,
                          kernel_size=ker_size, padding=pad).to(device)
            )
        
        self.layer_path.append(
            nn.Conv2d(in_channels=curr_N, out_channels=curr_N, 
                      kernel_size=ker_size, padding=pad).to(device)
        )


        for i in range(n_stages):
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N * 2, kernel_size=ker_size, padding=pad).cuda())
            curr_N *= 2
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())
        

        for i in range(n_stages):
            self.layer_path.append(nn.UpsamplingNearest2d(scale_factor=2).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N // 2, kernel_size=ker_size, padding=pad).cuda())

            curr_N = curr_N // 2

            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N * 2, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())


        for m in self.layer_path:
            if isinstance(m, CoordConv):
                nn.init.kaiming_normal_(m.conv2d_im.weight, mode='fan_in', nonlinearity='relu')  
                nn.init.kaiming_normal_(m.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
    
    def forward(self, x, cell_bg_coord):

        n_l = 0
        x_bridged = []
        if self.use_coordconv:
            x = F.elu(list(self.layer_path)[n_l](x, cell_bg_coord))
        else:
            x = F.elu(list(self.layer_path)[n_l](x))
        
        n_l += 1
        x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1

        x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                # this check is not used 
                if isinstance(list(self.layer_path)[n_l], CoordConv):
                    x = F.elu(list(self.layer_path)[n_l](x, cell_bg_coord))
                else:
                    x = F.elu(list(self.layer_path)[n_l](x))
                
                n_l += 1
                if n == 2 and i < self.n_stages - 1:
                    x_bridged.append(x)
        
        for i in range(self.n_stages):
            for n in range(4):
                if isinstance(list(self.layer_path)[n_l], CoordConv):
                    x = F.elu(list(self.layer_path)[n_l](x, cell_bg_coord))
                else:
                    x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 1:
                    x = torch.cat([x, x_bridged.pop()], 1) # concatenate from side
        
        return x
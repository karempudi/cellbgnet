import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.optimizer import Optimizer


def add_coord(input, field_xy, camera_chip_size):
    """ concatenate global coordinate channels to the input data
    Parameters
    ----------
    input:
         tensors with shape [batchsize, channel, height, width]
    field_xy:
        [xstart, xend, ystart, yend], the global position of the input sub-area image in the big aberration map.
        should satisfies this relationship: xstart - xend + 1 = input.width
    camera_chip_size:
        [sizex sizey], the size of the camera chip size,  sizex corresponds to column, sizey corresponds to row.
    """
    x_start = field_xy[0].float()
    y_start = field_xy[2].float()
    x_end = field_xy[1].float()
    y_end = field_xy[3].float()

    batch_size = input.size()[0]
    x_dim = input.size()[3]
    y_dim = input.size()[2]

    x_step = 1 / (camera_chip_size[0] - 1)
    y_step = 1 / (camera_chip_size[1] - 1)

    xx_range = torch.arange(x_start / (camera_chip_size[0] - 1), x_end / (camera_chip_size[0] - 1) + 1e-6, step=x_step,
                            dtype=torch.float32).repeat([y_dim, 1]).reshape([1, y_dim, x_dim])

    xx_range = xx_range.repeat_interleave(repeats=batch_size, dim=0).reshape([batch_size, 1, y_dim, x_dim])

    yy_range = torch.arange(y_start / (camera_chip_size[1] - 1), y_end / (camera_chip_size[1] - 1) + 1e-6, step=y_step,
                            dtype=torch.float32).repeat([x_dim, 1]).transpose(1, 0).reshape([1, y_dim, x_dim])

    yy_range = yy_range.repeat_interleave(repeats=batch_size, dim=0).reshape([batch_size, 1, y_dim, x_dim])

    xx_range = xx_range.cuda()
    yy_range = yy_range.cuda()

    ret = torch.cat([input, xx_range, yy_range], dim=1)

    return ret


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
        # n_in channels is 2 if you you xy or 3 if you also add cell based 
        # field
        self.conv2d_coord = nn.Conv2d(in_channels=2, out_channels=out_channels,
                                      kernel_size=1, padding=0)
        
    def forward(self, input, field_xy, camera_chip_size):
        y = add_coord(input, field_xy, camera_chip_size)
        ret_1 = self.conv2d_im(y[:, 0:-2])
        ret_2 = self.conv2d_coord(y[:, -2:])
        return ret_1 + ret_2


class OutnetBGConv(nn.Module):
    """
    Output from the U-net parts are transformed to the required number
    for calculating loss or just inferring
    """
    def __init__(self, n_filters, pred_sig=False, pred_bg=False, 
                 pad=1, ker_size=3, use_coordconv=False, device='cpu'):
        super(OutnetBGConv, self).__init__()

        self.pred_bg = pred_bg
        self.pred_sig = pred_sig
        self.use_coordconv = use_coordconv
        self.device = device

        if self.use_coordconv:
            self.p_out1 = CoordConv(in_channels=n_filters, out_channels=n_filters, 
                                    kernel_size=ker_size, padding=pad).to(device)
            self.p_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1, 
                                    kernel_size=1, padding=0).to(device)
            self.xyzi_out1 = CoordConv(in_channels=n_filters, out_channels=n_filters,
                                    kernel_size=ker_size, padding=pad).to(device)
            self.xyzi_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4,
                                    kernel_size=1, padding=0).to(device)
            
            
            nn.init.kaiming_normal_(self.p_out1.conv2d_im.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.p_out1.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='sigmoid')
            nn.init.constant_(self.p_out2.bias, -6.)  # -6

            nn.init.kaiming_normal_(self.xyzi_out1.conv2d_im.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.xyzi_out1.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.xyzi_out2.weight, mode='fan_in', nonlinearity='tanh')
            nn.init.zeros_(self.xyzi_out2.bias)


            if self.pred_sig:
                self.xyzis_out1 = CoordConv(in_channels=n_filters, out_channels=n_filters,
                                            kernel_size=ker_size, padding=pad).to(device)
                self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, 
                                            kernel_size=1, padding=0).to(device)

                nn.init.kaiming_normal_(self.xyzis_out1.conv2d_im.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.xyzis_out1.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.xyzis_out2.bias)

            if self.pred_bg:
                self.bg_out1 = CoordConv(in_channels=n_filters, out_channels=n_filters,
                                         kernel_size=ker_size, padding=pad).to(device)
                self.bg_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1,
                                         kernel_size=1, padding=0).to(device)

                nn.init.kaiming_normal_(self.bg_out1.conv2d_im.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.bg_out1.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.bg_out2.weight, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.bg_out2.bias)

        else:

            self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, 
                                    kernel_size=ker_size, padding=pad).to(device)
            self.p_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1,
                                    kernel_size=1, padding=0).to(device)
            self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, 
                                       kernel_size=ker_size, padding=pad).to(device)
            self.xyzi_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4,
                                       kernel_size=1, padding=0).to(device)

            nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='sigmoid')
            nn.init.constant_(self.p_out2.bias, -6.)  # -6

            nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.xyzi_out2.weight, mode='fan_in', nonlinearity='tanh')
            nn.init.zeros_(self.xyzi_out2.bias)

            if self.pred_sig:
                self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, 
                                            kernel_size=ker_size, padding=pad).to(device)
                self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4,
                                            kernel_size=1, padding=0).to(device)

                nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.xyzis_out2.bias)

            if self.pred_bg:
                self.bg_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, 
                                         kernel_size=ker_size, padding=pad).to(device)
                self.bg_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1, 
                                         kernel_size=1, padding=0).to(device)

                nn.init.kaiming_normal_(self.bg_out1.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.bg_out2.weight, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.bg_out2.bias)


    def forward(self, x, field_xy, camera_chip_size):
        outputs = {}

        if self.use_coordconv:
            p = F.elu(self.p_out1(x, field_xy, camera_chip_size))
            outputs['p'] = self.p_out2(p)

            xyzi = F.elu(self.xyzi_out1(x, field_xy, camera_chip_size))
            outputs['xyzi'] = self.xyzi_out2(xyzi)

            if self.pred_sig:
                xyzis = F.elu(self.xyzis_out1(x, field_xy, camera_chip_size))
                outputs['xyzi_sig'] = self.xyzis_out2(xyzis)

            if self.pred_bg:
                bg = F.elu(self.bg_out1(x, field_xy, camera_chip_size))
                outputs['bg'] = self.bg_out2(bg)
        else:
            p = F.elu(self.p_out1(x))
            outputs['p'] = self.p_out2(p)

            xyzi = F.elu(self.xyzi_out1(x))
            outputs['xyzi'] = self.xyzi_out2(xyzi)

            if self.pred_sig:
                xyzis = F.elu(self.xyzis_out1(x))
                outputs['xyzi_sig'] = self.xyzis_out2(xyzis)

            if self.pred_bg:
                bg = F.elu(self.bg_out1(x))
                outputs['bg'] = self.bg_out2(bg)

        return outputs


class UnetBGConv(nn.Module):
    """
    Used for both frame analysis and contex module.. In our
    case, context module is just an added U-net on top, so that
    the model becomes double U-nets, concatenated.    
    Cell Bg coordinate can be added on top if you set use_coordconv=True 
    """
    def __init__(self, n_inp, n_filters=64, n_stages=5, pad=1, ker_size=3,
                  use_coordconv=False, device='cpu'):
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
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=2, stride=2, padding=0).to(device))
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N * 2, kernel_size=ker_size, padding=pad).to(device))
            curr_N *= 2
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).to(device))
        

        for i in range(n_stages):
            self.layer_path.append(nn.UpsamplingNearest2d(scale_factor=2).to(device))
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N // 2, kernel_size=ker_size, padding=pad).to(device))

            curr_N = curr_N // 2

            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N * 2, out_channels=curr_N, kernel_size=ker_size, padding=pad).to(device))
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).to(device))


        for m in self.layer_path:
            if isinstance(m, CoordConv):
                nn.init.kaiming_normal_(m.conv2d_im.weight, mode='fan_in', nonlinearity='relu')  
                nn.init.kaiming_normal_(m.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
    
    def forward(self, x, field_xy, camera_chip_size):

        n_l = 0
        x_bridged = []
        if self.use_coordconv:
            x = F.elu(list(self.layer_path)[n_l](x, field_xy, camera_chip_size))
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
                    x = F.elu(list(self.layer_path)[n_l](x, field_xy, camera_chip_size))
                else:
                    x = F.elu(list(self.layer_path)[n_l](x))
                
                n_l += 1
                if n == 2 and i < self.n_stages - 1:
                    x_bridged.append(x)
        
        for i in range(self.n_stages):
            for n in range(4):
                if isinstance(list(self.layer_path)[n_l], CoordConv):
                    x = F.elu(list(self.layer_path)[n_l](x, field_xy, camera_chip_size))
                else:
                    x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 1:
                    x = torch.cat([x, x_bridged.pop()], 1) # concatenate from side
        
        return x
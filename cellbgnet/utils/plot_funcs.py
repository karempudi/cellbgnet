import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import math
from typing import Optional
import numpy as np

from cellbgnet.simulation.psf_kernel import SMAPSplineCoefficient

def animate_psf(one_psf):
    fig, ax = plt.subplots()
    ims = []
    for i in range(one_psf.shape[0]):
        im = ax.imshow(one_psf[i], animated=True)
        if i == 0:
            ax.imshow(one_psf[i])
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims,  interval=50, blit=True, repeat_delay=1000)
    plt.show()
    return ani


def plot_psf(calib_file, z_range=500.0, plot_step=25.0, img_size=41,
            dz=25.0, pixel_size_xy=[65.0, 65.0], animate=True,
            photon_counts=10000.0, bg_photons=100.0):
    """
    Plot a point spread function from the spline model
    
    Arguments:
    ------------
        calib_file: spline model file from SMAP

        z_range (float, in nms): z_range over which you want to simulate the PSF

        plot_step (float in nms): increments in steps at which psf will be simulated

        img_size (odd number): size of the image that will have dots at the center pixel
        
        animate (bool): Also animate the simulated PSF

    """

    psf = SMAPSplineCoefficient(calib_file).init_spline(
        xextent=[-0.5, img_size-0.5], yextent=[-0.5, img_size-0.5], 
        img_shape=[img_size, img_size], device='cpu',
        roi_size=None, roi_auto_center=None
    )
    
    psf.vx_size = torch.tensor([pixel_size_xy[0], pixel_size_xy[1], dz])

    z = torch.arange(-z_range, z_range+plot_step, plot_step)

    n_planes = len(z)
    print(n_planes)

    xyz = torch.zeros((len(z), 3))
    xyz[:, 0] = img_size // 2
    xyz[:, 1] = img_size // 2
    xyz[:, 2] = z
    phot = photon_counts * torch.ones((n_planes,))
    bg = bg_photons * torch.ones((n_planes,))

    _, rois = psf.crlb_sq(xyz, phot, bg)
    rois /= rois.sum(-1).sum(-1)[:, None, None]

    n_cols = 6
    n_rows = math.ceil(n_planes/n_cols)
    print(f"N rows: {n_rows} and N cols: {n_cols}")
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i in range(n_planes):
        ax[i//n_cols, i%n_cols].imshow(rois[i])
        ax[i//n_cols, i%n_cols].set_title(f"{z[i]} (nm)")

    for i in range(n_planes, n_rows * n_cols):
        ax[i//n_cols, i%n_cols].remove()

    plt.show()
    return rois


def plot_od(od, label=None, col=None):
    """Produces a line plot from a ordered dictionary as used to store training process in the Model class
    
    Parameters
    ----------
    od: OrderedDict of floats
        DECODE model
    label: str
        Label
    col: 'str'
        Color
    """
    plt.plot(*zip(*sorted(od.items())), label=label, color=col)

def plot_train_record(model):
    plt.figure(figsize=(9,6),constrained_layout=True)
    plt.subplot(4,4,1);plot_od(model.recorder['rmse_lat']);plt.xlabel('iterations');plt.ylabel('RMSE_Lateral')
    plt.subplot(4,4,2);plot_od(model.recorder['rmse_ax']);plt.xlabel('iterations');plt.ylabel('RMSE_Axial')
    plt.subplot(4,4,3);plot_od(model.recorder['rmse_vol']);plt.xlabel('iterations');plt.ylabel('RMSE_Voxel')
    plt.subplot(4,4,4);plot_od(model.recorder['eff_lat']);plt.xlabel('iterations');plt.ylabel('lateral efficiency')
    plt.subplot(4,4,5);plot_od(model.recorder['eff_3d']);plt.xlabel('iterations');plt.ylabel('3D efficiency')
    plt.subplot(4,4,6);plot_od(model.recorder['recall']);plt.xlabel('iterations');plt.ylabel('recall')
    plt.subplot(4,4,7);plot_od(model.recorder['precision']);plt.xlabel('iterations');plt.ylabel('precision')
    plt.subplot(4,4,8);plot_od(model.recorder['jaccard']);plt.xlabel('iterations');plt.ylabel('jaccard')
    plt.subplot(4,4,9);plot_od(model.recorder['cost_hist']);plt.xlabel('iterations');plt.ylabel('cost')
    plt.subplot(4,4,10);plot_od(model.recorder['rmse_x']);plt.xlabel('iterations');plt.ylabel('RMSE_x')
    plt.subplot(4,4,11);plot_od(model.recorder['rmse_y']);plt.xlabel('iterations');plt.ylabel('RMSE_y')
    plt.subplot(4,4,12);plot_od(model.recorder['count_loss']);plt.xlabel('iterations');plt.ylabel('Count Loss')
    plt.subplot(4,4,13);plot_od(model.recorder['loc_loss']);plt.xlabel('iterations');plt.ylabel('Localization Loss')
    plt.subplot(4,4,14);plot_od(model.recorder['bg_loss']);plt.xlabel('iterations');plt.ylabel('Background Loss')
    plt.subplot(4,4,15);plot_od(model.recorder['P_locs_error']);plt.xlabel('iterations');plt.ylabel('Cross entropy loss')
    # plt.subplots_adjust(wspace=0.5,hspace=0.5)
    # plt.tight_layout()
    plt.show()

def plot_preds_distribution(preds,preds_final):
    fig,axes = plt.subplots(2,2)
    axes[0,0].hist(np.array(preds)[:, 6], bins=50)
    axes[0,0].axvspan(np.array(preds_final)[:, 6].min(),np.array(preds_final)[:, 6].max(),color='green', alpha=0.1)
    axes[0,0].set_xlabel(r'$nms-p$')
    axes[0,0].set_ylabel('counts')

    axes[0,1].hist(np.array(preds)[:, 7], bins=50)
    axes[0,1].axvspan(0, np.array(preds_final)[:, 7].max(),color='green', alpha=0.1)
    axes[0,1].set_xlabel(r'$\sigma_x$ [nm]')
    axes[0,1].set_ylabel('counts')

    axes[1,0].hist(np.array(preds)[:, 8], bins=50)
    axes[1,0].axvspan(0, np.array(preds_final)[:, 8].max(),color='green', alpha=0.1)
    axes[1,0].set_xlabel(r'$\sigma_y$ [nm]')
    axes[1,0].set_ylabel('counts')

    axes[1,1].hist(np.array(preds)[:, 9], bins=50)
    axes[1,1].axvspan(0, np.array(preds_final)[:, 9].max(),color='green', alpha=0.1)
    axes[1,1].set_xlabel(r'$\sigma_z$ [nm]')
    axes[1,1].set_ylabel('counts')
    plt.tight_layout()
    plt.show()
    return fig,axes

# From original decode
def connect_point_set(set0, set1, threeD=False, ax=None):
    """
    Plots the connecting lines between the set0 and set1 in 2D.

    Args:
        set0:  torch.Tensor / np.array of dim N x 2
        set1:  torch.Tensor / np.array of dim N x 2
        threeD (bool): plot / connect in 3D
        ax:  axis where to plot

    Returns:
        nothing
    """
    if ax is None:
        ax = plt.gca()

    if threeD:
        for i in range(set0.size(0)):
            ax.plot3D([set0[i, 0], set1[i, 0]], [set0[i, 1], set1[i, 1]], [set0[i, 2], set1[i, 2]],
                      'orange')
    else:
        for i in range(set0.size(0)):
            ax.plot([set0[i, 0], set1[i, 0]], [set0[i, 1], set1[i, 1]], 'orange')

# From original decode
class PlotFrame:
    def __init__(self, frame: torch.Tensor, extent: Optional[tuple] = None, clim=None,
                 plot_colorbar: bool = False, axes_order: Optional[str] = None):
        """
        Plots a frame.

        Args:
            frame: frame to be plotted
            extent: specify frame extent, tuple ((x0, x1), (y0, y1))
            clim: clim values
            plot_colorbar: plot the colorbar
            axes_order: order of axis. Either default order (None) or 'future'
             (i.e. future version of decode in which we will swap axes).
             This is only a visual effect and does not change the storage scheme of the EmitterSet

        """

        self.frame = frame.detach().squeeze()
        self.extent = extent
        self.clim = clim
        self.plot_colorbar = plot_colorbar
        self._axes_order = axes_order

        assert self._axes_order is None or self._axes_order == 'future'

        if self._axes_order is None:
            self.frame.transpose_(-1, -2)

    def plot(self) -> plt.axis:
        """
        Plot the frame. Note that according to convention we need to transpose the last two axis.
        """
        if self.extent is None:
            plt.imshow(self.frame.numpy(), cmap='gray')
        else:
            plt.imshow(self.frame.numpy(), cmap='gray', extent=(
                self.extent[0][0],
                self.extent[0][1],
                self.extent[1][1],
                self.extent[1][0]))

        plt.gca().set_aspect('equal', adjustable='box')
        if self.clim is not None:
            plt.clim(self.clim[0], self.clim[1])
            # safety measure
        if self.plot_colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel('x')
        plt.ylabel('y')

        return plt.gca()

# Modified from original decode
class PlotCoordinates:
    _labels_default = ('Target', 'Output')
    
    def __init__(self, 
                pos_tar=None, phot_tar=None,
                pos_out=None, phot_out=None,
                extent_limit=None,
                match_lines=False,
                labels=None,
                axes_order: Optional[str] = None):
        """
        Plots points in 2D projection, only x and y 
        
        Arguments:
        ----------
            pos_tar:
            phot_tar:
            pos_out:
            phot_out:
            extent_limit:
            match_lines:
            labels:
            axes_order:
        
        """
        self.pos_tar = pos_tar
        self.phot_tar = phot_tar
        self.pos_out = pos_out
        self.phot_out = phot_out
        self.extent_limit = extent_limit
        self.match_lines = match_lines
        self.labels = labels if labels is not None else self._labels_default
        self._axes_order = axes_order
        
        # color code parameters
        self.tar_marker = 'ro'
        self.tar_cmap = 'winter'
        self.out_marker = 'bx'
        self.out_cmap = 'viridis'
        
        assert self._axes_order is None or self._axes_order == 'future'
        
    def plot(self):
        
        def plot_xyz(pos, marker, color, label):
            if self._axes_order == 'future':
                pos = pos[:, [1, 0, 2]]
            plt.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(),
                       marker=marker, c=color, facecolors='none', label=label)
        
        def plot_xyz_phot(pos, phot, marker, cmap, label):
            if self._axes_order == 'future':
                pos = pos[:, [1, 0, 2]]
            plt.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(), c=phot.numpy(),
                        marker=marker, facecolors='none', cmap=cmap, label=label)
        
        
        if self.pos_tar is not None:
            if self.phot_tar is not None:
                plot_xyz_phot(self.pos_tar, self.phot_tar, self.tar_marker[1], self.tar_cmap,
                             self.labels[0])
            else:
                plot_xyz(self.pos_tar, self.tar_marker[1], self.tar_marker[0], self.labels[0])
        
        if self.pos_out is not None:
            if self.phot_out is not None:
                plot_xyz_phot(self.pos_out, self.phot_out, self.out_marker[1], self.out_cmap,
                             self.labels[1])
            else:
                plot_xyz(self.pos_out, self.out_marker[1], self.out_marker[0], self.labels[1])
                
        if self.pos_tar is not None and self.pos_out is not None and self.match_lines:
            connect_point_set(self.pos_tar, self.pos_out, threeD=False)
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        ax_ylimits = ax.get_ylim()
        if ax_ylimits[0] <= ax_ylimits[1]:
            ax.set_ylim(ax_ylimits[::-1])  # invert the axis

        if self._axes_order is None:
            plt.xlabel('x')
            plt.ylabel('y')
        else:
            plt.xlabel('y')
            plt.ylabel('x')

        if self.extent_limit is not None:
            plt.xlim(*self.extent_limit[0])
            plt.ylim(*self.extent_limit[1][::-1])  # reverse tuple order
        
        plt.legend()

        return plt.gca()

# From original decode, only sligthly modified
class PlotFrameCoord(PlotCoordinates, PlotFrame):

    def __init__(self, frame,
                 pos_tar=None, phot_tar=None,
                 pos_out=None, phot_out=None,
                 extent=None, coord_limit=None,
                 norm=None, clim=None,
                 match_lines=False, labels=None,
                 plot_colorbar_frame: bool = False,
                 axes_order: Optional[str] = None):

        PlotCoordinates.__init__(self,
                                 pos_tar=pos_tar,
                                 phot_tar=phot_tar,
                                 pos_out=pos_out,
                                 phot_out=phot_out,
                                 extent_limit=coord_limit,
                                 match_lines=match_lines,
                                 labels=labels,
                                 axes_order=axes_order)

        PlotFrame.__init__(self, frame, extent, clim,
                           plot_colorbar=plot_colorbar_frame, axes_order=axes_order)

    def plot(self):
        PlotFrame.plot(self)
        PlotCoordinates.plot(self)
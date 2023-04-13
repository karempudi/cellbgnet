import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import math

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

    plt.tight_layout()
    plt.show()


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
    plt.subplot(4,3,1);plot_od(model.recorder['rmse_lat']);plt.xlabel('iterations');plt.ylabel('RMSE_Lateral')
    plt.subplot(4,3,2);plot_od(model.recorder['rmse_ax']);plt.xlabel('iterations');plt.ylabel('RMSE_Axial')
    plt.subplot(4,3,3);plot_od(model.recorder['rmse_vol']);plt.xlabel('iterations');plt.ylabel('RMSE_Voxel')
    plt.subplot(4,3,4);plot_od(model.recorder['eff_lat']);plt.xlabel('iterations');plt.ylabel('lateral efficiency')
    plt.subplot(4,3,5);plot_od(model.recorder['eff_3d']);plt.xlabel('iterations');plt.ylabel('3D efficiency')
    plt.subplot(4,3,6);plot_od(model.recorder['recall']);plt.xlabel('iterations');plt.ylabel('recall')
    plt.subplot(4,3,7);plot_od(model.recorder['precision']);plt.xlabel('iterations');plt.ylabel('precision')
    plt.subplot(4,3,8);plot_od(model.recorder['jaccard']);plt.xlabel('iterations');plt.ylabel('jaccard')
    plt.subplot(4,3,9);plot_od(model.recorder['cost_hist']);plt.xlabel('iterations');plt.ylabel('cost')
    plt.subplot(4,3,10);plot_od(model.recorder['rmse_x']);plt.xlabel('iterations');plt.ylabel('RMSE_x')
    plt.subplot(4,3,11);plot_od(model.recorder['rmse_y']);plt.xlabel('iterations');plt.ylabel('RMSE_y')
    # plt.subplots_adjust(wspace=0.5,hspace=0.5)
    # plt.tight_layout()
    plt.show(block=True)

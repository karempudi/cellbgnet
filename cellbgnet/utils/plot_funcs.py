import matplotlib.pyplot as plt


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

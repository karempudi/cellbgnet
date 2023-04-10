import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as func
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from scipy.spatial.distance import cdist
import os


def recognition(model, eval_imgs_all, batch_size, use_tqdm, nms, pixel_nm, plot_num,
            win_size=128, padding=True, candidate_threshold=0.3,
            nms_threshold=0.3):
    
    """
    Analyze the SMLM model from cellbgnet model or decode version of this

    Arguments:
    -----------
        model : an object of class CellBGModel

        eval_imgs_all (np.ndarray): all the images that need to be used for evalutation 
                                    (to calculate metrics)
        
        batch_size (int): Number of images in each batch, (set small to save GPU memory)

        use_tqdm (bool): Use progress bar

        nms (bool): If False, only use a simple threshold to filter P channel
                        to get the deterministic pixels.
                    If True, add the values from the 4 adjacent pixels to 
                        local maximums and then filter with the threshold.
        pixel_nm: Pixel size in nms [X_pixelsize, Y_pixelsize] 

        plot_num (int or None): Chose a specific frame to return the output

        win_size: TODO will come back to this when you understand it fully

        padding (bool): If padding=True, this will cut a larger area (20 pixels) than win_size and
                        traverse with overlap to avoid error from incomplete PSFs at the margin.
        
        candidate_threshold (float): In the probability channel, only pixels > threshold will be
                        treated as candidates for local maximum searching.

        nms_threshold (float): This works as a threshold to filter P channel to get the 
                               deterministic pixels

    Returns:
    -----------

    """
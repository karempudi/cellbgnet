import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation
import pathlib
from pathlib import Path


def overlay_cell_boundray_on_fluor(fluor_img, mask_img, contrast_factor=3, color=(1, 0, 0), mode='outer'):
    """
    Function to overlay boundaries of the cell segmentation mask on top of fluor img

    """
    edges_img = segmentation.mark_boundaries(fluor_img*3 / fluor_img.max(), mask_img, color=color, mode=mode)
    plt.figure()
    plt.imshow(edges_img)
    plt.show()

def sample_bg_photons():
    pass
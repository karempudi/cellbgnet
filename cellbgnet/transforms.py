
import pathlib
from pathlib import Path
import numpy as np
from skimage.transform import AffineTransform, warp
from scipy.io import loadmat
import matplotlib.pyplot as plt


def phase_to_flour(phase_img, out_shape, transformation_matrix, plot=False):
    """
    Transform the phase img to the same shape as that of fluor image
    using the transformation matrix

    Arguments:
    -----------
        phase_img: phase contrast image (typically 2000x2000 image)

        out_shape:  H x W of the image size to which phase_img will
                          be transformed.
                    
        transformation_matrix (str or np.ndarray): a string path of the matrix or
                         an numpy array that is already loaded with values. If string,
                         then will load the file from disk
        
        plot (bool): If true will plot transformed output along with the input

    """
    if isinstance(transformation_matrix, str):
        transformation_matrix_path = Path(transformation_matrix)
        transformation_mat = loadmat(transformation_matrix_path)['transformationMatrix'].T
    elif isinstance(transformation_matrix, pathlib.Path):
        transformation_mat = loadmat(transformation_matrix)['transformationMatrix'].T
    elif isinstance(transformation_matrix, np.ndarray):
        transformation_mat = transformation_matrix

    affine_fluor = AffineTransform(transformation_mat)
    phase_out = warp(phase_img, affine_fluor, output_shape=out_shape, preserve_range=True)

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(phase_img)
        ax[0].set_title('Phase image input')
        ax[1].imshow(phase_out)
        ax[1].set_title('Phase image after transformation')

    return phase_out




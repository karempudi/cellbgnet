import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib
from pathlib import Path
import seaborn as sns
import scipy.stats
import edt
from skimage import segmentation
from skimage.io import imread
import json
import pickle
from tqdm import tqdm

sns.set_style('white')


##################################################
############ Chromosomal data functions ##########
##################################################


def chromo_mean_var_bg_outside(fluor_img, cellseg_mask, dilate=True, roi=None,
                 plot=False, dilate_px=1, return_alpha_beta=False):
    """
    Function to evaluate mean and variance of the pixels outside the cells
    and fit gamma distribution to the pixels

    Arguments:
    ----------
        fluor_img (np.ndarray): fluorescence image
        cellseg_mask (np.ndarray): corresponding cell mask for the fluorescence image
        dilate (bool): Should you dilate the cell mask a bit, defualt is 1 in dilate_px
        roi (list of 4 ints): rchw of the ROI you want to calculate incase cell masks
                              are not that great in some regions
        plot (bool): Plot the fitted gamma distribution

        return_alpha_beta (bool): function returns alpha and beta instead of mean and
                    variance of the gamma distribution

    Returns:
    ----------
        mean, variance of the fitted gamma distribution to the pixel values that
        are outside the cells
    """
    if roi is not None:
        cellseg_mask = cellseg_mask[roi[0]: roi[0] + roi[2], roi[1]: roi[1] + roi[3]]
        fluor_img = fluor_img[roi[0]: roi[0] + roi[2], roi[1]: roi[1] + roi[3]]

    if dilate:
        cellseg_mask = segmentation.expand_labels(cellseg_mask, distance=dilate_px)
    binary_mask = cellseg_mask == 0
    
    img_outside_cells = np.zeros_like(fluor_img)

    outside_inds = np.where(binary_mask == 1)
    img_outside_cells[outside_inds[0], outside_inds[1]] = fluor_img[outside_inds[0], outside_inds[1]]

    only_pixels_outside = img_outside_cells[outside_inds[0], outside_inds[1]].ravel()

    # remove outliers in the outside pixels at 99% 
    collect_bg_only = only_pixels_outside[np.where(only_pixels_outside < np.percentile(only_pixels_outside, 99))]

    # now fit a gamma distribution and return the mean and variance of this fitted distribution
    fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(collect_bg_only, floc=0.0)
    low, high = collect_bg_only.min(), collect_bg_only.max()

    if plot:
        fig, ax = plt.subplots()
        ax.hist(collect_bg_only, bins=np.arange(low, high), histtype='step', label='data')
        ax.hist(np.random.gamma(shape=fit_alpha, scale=fit_beta, size=len(collect_bg_only)+0),
                        bins=np.arange(low, high), histtype='step', label='fit')
        ax.set_xlabel('Chip background (ADU)', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        plt.legend()
        plt.show()
    
    if return_alpha_beta:
        return fit_alpha, fit_beta
    else:
        return fit_alpha * fit_beta, fit_alpha * fit_beta * fit_beta

def chromo_mean_var_bg_inside(fluor_img, cellseg_mask, bg_cutoff_percentile=75, dilate=True,
                    roi=None, plot=False, dilate_px=1):

    """
    Function that will get mean and varaince of background, argumnets are more or less 
    same as that of chromo_mean_var_bg_outside function.
    """
    if roi is not None:
        cellseg_mask = cellseg_mask[roi[0]: roi[0] + roi[2], roi[1]: roi[1] + roi[3]]
        fluor_img = fluor_img[roi[0]: roi[0] + roi[2], roi[1]: roi[1] + roi[3]]

    if dilate:
        cellseg_mask = segmentation.expand_labels(cellseg_mask, distance=dilate_px)
    
    binary_mask = cellseg_mask > 0
    inside_inds = np.where(binary_mask == 1)
    img_inside_cells = np.zeros_like(fluor_img)
    img_inside_cells[inside_inds[0], inside_inds[1]] = fluor_img[inside_inds[0], inside_inds[1]]

    only_pixels_inside = img_inside_cells[inside_inds[0], inside_inds[1]].ravel()

    # remove outliers in the inside pixels at 99%
    removing_outliers = only_pixels_inside[np.where(only_pixels_inside < np.percentile(only_pixels_inside, 99))]
    # collect only background using percentile cutoff
    collect_bg_only = removing_outliers[np.where(removing_outliers < np.percentile(removing_outliers, bg_cutoff_percentile))]
    #collect_bg_only = removing_outliers
    collect_dots_only = removing_outliers[np.where(removing_outliers > np.percentile(removing_outliers, bg_cutoff_percentile))]
    # now fit a gamma distribution and return the mean of the gamma distribution
    #print("Max value inside: ", max_values_inside)
    fit_alpha_bg, fit_loc_bg, fit_beta_bg = scipy.stats.gamma.fit(collect_bg_only, floc=0)
    fit_alpha_dots, fit_loc_dots, fit_beta_dots = scipy.stats.gamma.fit(collect_dots_only, floc=0)
    low_bg,  high_bg = collect_bg_only.min(), collect_bg_only.max()
    low_dots, high_dots = collect_dots_only.min(), collect_dots_only.max()
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].hist(collect_bg_only, bins=np.linspace(low_bg, high_bg), histtype='barstacked', label='data')
        ax[0].hist(np.random.gamma(shape=fit_alpha_bg, scale=fit_beta_bg, size=len(collect_bg_only)+0),
                        bins=np.linspace(low_bg, high_bg), histtype='step', label='fit')
        ax[0].legend()
        ax[0].set_title('Bg only')
        ax[1].hist(collect_dots_only, bins=np.linspace(low_dots, high_dots), histtype='step', label='data')
        ax[1].hist(np.random.gamma(shape=fit_alpha_dots, scale=fit_beta_dots, size=len(collect_dots_only)+0),
                        bins=np.linspace(low_dots,  high_dots), histtype='step', label='fit')
        ax[1].set_title('Dots only')
        ax[1].legend()
        plt.show()
    
    print(f"Max bg value: {collect_bg_only.max()}, max dot vale cell: {collect_dots_only.max()}")
    print(f"Mean of gamma dist bg: {fit_alpha_bg * fit_beta_bg}, variance: {fit_alpha_bg * fit_beta_bg * fit_beta_bg}")

    return fit_alpha_bg * fit_beta_bg, fit_alpha_bg * fit_beta_bg * fit_beta_bg



def chromo_edt_mean_variance_inside(fluor_img, cellseg_mask, bg_cutoff_percentile=75, dilate=True,
                        dilate_px=1, roi=None, maximum_edt=7, plot=False, return_values=False):
    if roi is not None:
        cellseg_mask = cellseg_mask[roi[0]: roi[0] + roi[2], roi[1]: roi[1] + roi[3]]
        fluor_img = fluor_img[roi[0]: roi[0] + roi[2], roi[1]: roi[1] + roi[3]]

    if dilate:
        cellseg_mask = segmentation.expand_labels(cellseg_mask, distance=dilate_px)
    
    binary_mask = cellseg_mask > 0
    inside_inds = np.where(binary_mask == 1)
    img_inside_cells = np.zeros_like(fluor_img)
    img_inside_cells[inside_inds[0], inside_inds[1]] = fluor_img[inside_inds[0], inside_inds[1]]

    only_pixels_inside = img_inside_cells[inside_inds[0], inside_inds[1]].ravel()
    #print(only_pixels_inside)

    # removing outliers in the inside pixels at 99%
    removing_outliers = only_pixels_inside[np.where(only_pixels_inside < np.percentile(only_pixels_inside, 99))]
    # collect background values
    collect_bg_only = removing_outliers[np.where(removing_outliers < np.percentile(removing_outliers, bg_cutoff_percentile))]

    cutoff_bg = np.percentile(removing_outliers, bg_cutoff_percentile)
    #print(cutoff_bg)

    # This not has only the background inside cells and not 
    # 
    bg_removed_dots = img_inside_cells.copy()
    bg_removed_dots[bg_removed_dots > cutoff_bg] = 0.0

    if plot:
        plt.figure()
        plt.imshow(bg_removed_dots)
        plt.title('Background removed dots ... ')
        plt.show()

    dists = edt.edt(cellseg_mask)
    min_edt, max_edt = int(dists.min()), int(min(dists.max(), maximum_edt))
    mean_noise_map = {}
    stddev_noise_map = {}
    counts_noise_map = {}
    edt_noise_map = {}
    edt_values = {}
    fitted_dist_params = {}
    for i in range(min_edt+1, max_edt+1, 1):
        edt_i = bg_removed_dots[dists==i]
        edt_i = edt_i[edt_i > 0]

        if len(edt_i) > 0:
            fit_alpha_edt, fit_loc_edt, fit_beta_edt = scipy.stats.gamma.fit(edt_i, floc=-1.5)
            fitted_dist_params[i] = {
                'alpha': fit_alpha_edt,
                'loc': fit_loc_edt, 
                'beta': fit_beta_edt,
            }

            mean_noise_map[i] = np.mean(edt_i)
            stddev_noise_map[i] = np.std(edt_i)
            counts_noise_map[i] =  len(edt_i)

        if return_values == True:
            edt_values[i] = edt_i
        else:
            edt_values[i] = None

    edt_noise_map['mean'] = mean_noise_map
    edt_noise_map['stddev'] = stddev_noise_map
    edt_noise_map['counts'] = counts_noise_map
    edt_noise_map['values'] = edt_values
    edt_noise_map['fits'] = fitted_dist_params

    return edt_noise_map

def get_full_edt_maps(masks_dir, fluor_dir, save_filename=None,
                mask_fileformat='.png', fluor_fileformat='.tiff',
                roi=[170, 420, 720, 720], edt_min_max=[1, 7], bg_cutoff_percentile=75):
    """
    Gets edt maps that you can save and reload them back later.
    
    Save as pickle file if given a filename

    Arguments:
        roi: for bg, we use this roi rchw notation to get central region
                incase some cells are not segmented at the edges
    """
    if type(masks_dir) == str:
        masks_dir = Path(masks_dir)
    if type(fluor_dir) == str:
        fluor_dir = Path(fluor_dir)

    mask_filenames = sorted(list(masks_dir.glob('*' + mask_fileformat)))
    fluor_filenames = sorted(list(fluor_dir.glob('*' + fluor_fileformat)))
    
    # first we collect all the background values
    chip_bg_alphas = []
    chip_bg_betas = []
    edt_fit_pool_alphas = {}
    edt_fit_pool_betas  = {}
    for edt_val in range(edt_min_max[0], edt_min_max[1]+1):
        edt_fit_pool_alphas[edt_val] = []
        edt_fit_pool_betas[edt_val] = []
    
    for index in tqdm(range(len(mask_filenames))):
        mask_img = imread(mask_filenames[index])
        fluor_img = imread(fluor_filenames[index])
        alpha_bg, beta_bg = chromo_mean_var_bg_outside(fluor_img, mask_img, 
                            plot=False, roi=roi, return_alpha_beta=True)

        chip_bg_alphas.append(alpha_bg)
        chip_bg_betas.append(beta_bg)
        try:
            edt_data = chromo_edt_mean_variance_inside(fluor_img, mask_img, plot=False, return_values=False,
                                 bg_cutoff_percentile=bg_cutoff_percentile)
            for edt_val in range(edt_min_max[0], edt_min_max[1]):
                if edt_data['fits'][edt_val]['alpha'] != float('nan'):
                    edt_fit_pool_alphas[edt_val].append(edt_data['fits'][edt_val]['alpha'])
                if edt_data['fits'][edt_val]['beta'] != float('nan'):
                    edt_fit_pool_betas[edt_val].append(edt_data['fits'][edt_val]['beta'])
 
        except Exception as e:
            print(f"found some nan probably as file number: {index}")
            print(f"Exception: {e}")

    for key in list(edt_fit_pool_alphas.keys()):
        if len(edt_fit_pool_alphas[key]) != 0:
            edt_fit_pool_alphas[key] = np.mean(edt_fit_pool_alphas[key])
        else:
            del edt_fit_pool_alphas[key]
    for key in list(edt_fit_pool_betas.keys()):
        if len(edt_fit_pool_betas[key]) != 0:
            edt_fit_pool_betas[key] = np.mean(edt_fit_pool_betas[key])
        else:
            del edt_fit_pool_betas[key]
    
    edt_fit_pool_alphas[0] = np.mean(chip_bg_alphas)
    edt_fit_pool_betas[0] = np.mean(chip_bg_betas)
    for key in list(edt_fit_pool_alphas.keys()):
        if edt_fit_pool_alphas[key] == float('nan'):
            del edt_fit_pool_alphas[key]
            del edt_fit_pool_betas[key]
    
    if save_filename != None:
        with open(save_filename, "wb") as fp:
            pickle.dump({
                'alphas': edt_fit_pool_alphas,
                'betas': edt_fit_pool_betas
            }, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return edt_fit_pool_alphas, edt_fit_pool_betas
    


    # second we collect all the values from the cells


###################################################
############# Replisome data functions ############
###################################################



###################################################
############ Membrane dots functions ##############
###################################################
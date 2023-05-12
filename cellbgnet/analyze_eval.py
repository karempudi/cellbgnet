import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as func
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from scipy.spatial.distance import cdist
import os
import copy
import operator
import csv

from cellbgnet.utils.hardware import cpu, gpu
from cellbgnet.simulation.psf_kernel import SMAPSplineCoefficient


def flip_filt(filt):
    '''Returns filter flipped over x and y dimension'''
    return np.ascontiguousarray(filt[...,::-1,::-1])

def find_frame(molecule_list, frame_nbr):
    """find the molecule list index of the specified frame_number then plus 1 (the molecule list is sorted)"""
    list_index = None
    for i,molecule in enumerate(molecule_list):
        if molecule[1] > frame_nbr:
            list_index = i
            break
    return list_index


def infer_imgs(model, images, field_xy, batch_size=100, z_scale=10, int_scale=10, use_tqdm=False):
    """
    Performs inference for a given set of images.
    
    Parameters
    ----------
    model: Model
        An instance of CellBGModel
    images: numpy array
        Three dimensional array of SMLM images
    field_xy: a list of four number 
        corresponding to the field 
    batch_size: int
        Images are proccessed in batches of the given size. 
        When the images are large, the batch size has to be lowered to save GPU memory. 
    z_scale: float
        The model outputs z values between -1 and 1 that are rescaled.
    int_scale: float
        The model outputs photon values between 0 and 1 that are rescaled.
        
    Returns
    -------
    infs: dict
        Dictionary of arrays with the rescaled network outputs
    """

    with torch.no_grad():
        N = len(images)
        if N != 1:
            images = np.concatenate([images[1:2], images, images[-2:-1]], 0).astype('float32')
        else:
            images = np.concatenate([images, images, images], 0).astype('float32')

        if use_tqdm:
            tqdm_func = tqdm
        else:
            def tqdm_func(x):
                return x

        infs = {'Probs': [], 'XO': [], 'YO': [], 'ZO': [], 'Int': []}
        if model.psf_pred:
            infs['BG'] = []
        if model.sig_pred:
            infs['XO_sig'] = []
            infs['YO_sig'] = []
            infs['ZO_sig'] = []
            infs['Int_sig'] = []

        for i in tqdm_func(range(int(np.ceil(N / batch_size)))):
            p, xyzi, xyzi_sig, bg = model.inferring(X=gpu(images[i * batch_size:(i + 1) * batch_size + 2]),
                                                    field_xy=field_xy,
                                                    camera_chip_size=[model.data_generator.camera_chip_size[1],
                                                                      model.data_generator.camera_chip_size[0]])

            infs['Probs'].append(p[1:-1].cpu())  # index[1：-1] because the input stack was concatenated with two images
            infs['XO'].append(xyzi[1:-1, 0].cpu())
            infs['YO'].append(xyzi[1:-1, 1].cpu())
            infs['ZO'].append(xyzi[1:-1, 2].cpu())
            infs['Int'].append(xyzi[1:-1, 3].cpu())
            if model.sig_pred:
                infs['XO_sig'].append(xyzi_sig[1:-1, 0].cpu())
                infs['YO_sig'].append(xyzi_sig[1:-1, 1].cpu())
                infs['ZO_sig'].append(xyzi_sig[1:-1, 2].cpu())
                infs['Int_sig'].append(xyzi_sig[1:-1, 3].cpu())
            if model.psf_pred:  # rescale the psf estimation
                # TODO incase you don't see correct results, verify the math here
                # You will predict the values in ADU, so that you can put the 
                # images for comparision, 
                # 10 is important to divide cuz you multiplied to balance the loss function components
                infs['BG'].append(bg[1:-1].cpu() * model.psf_params['photon_scale'] / 10
                                  * model.data_generator.simulation_params['qe'] 
                                  / model.data_generator.simulation_params['e_per_adu'])
                

        for k in infs.keys():
            infs[k] = np.vstack(infs[k])

        # # scale the predictions
        # infs['ZO'] = z_scale * infs['ZO']
        # infs['Int'] = int_scale * infs['Int']
        # if model.sig_pred:
        #     infs['Int_sig'] = int_scale * infs['Int_sig']
        #     infs['ZO_sig'] = z_scale * infs['ZO_sig']
        return infs


def nms_sampling(res_dict, threshold=0.3, candi_thre=0.3, batch_size=500, nms=True, nms_cont=False):
    """Performs Non-maximum Suppression to obtain deterministic samples from the probabilities provided by the decode function. 
    
    Parameters
    ----------
    res_dict: dict
        Dictionary of arrays created with decode_func
    threshold: float
        Processed probabilities above this threshold are considered as final detections
    candi_thre:float
        Probabilities above this threshold are treated as candidates
    batch_size: int
        Outputs are proccessed in batches of the given size. 
        When the arrays are large, the batch size has to be lowered to save GPU memory. 
    nms: bool
        If False don't perform Non-maximum Suppression and simply applies a theshold to the probablities to obtain detections
    nms_cont: bool
        If true also averages the offset variables according to the probabilties that count towards a given detection
        
    Returns
    -------
    res_dict: dict
        Dictionary of arrays where 'Samples_ps' contains the final detections
    """

    res_dict['Probs_ps'] = res_dict['Probs'] + 0  # after nms, this is the final-probability
    res_dict['XO_ps'] = res_dict['XO'] + 0
    res_dict['YO_ps'] = res_dict['YO'] + 0
    res_dict['ZO_ps'] = res_dict['ZO'] + 0

    if nms:

        N = len(res_dict['Probs'])
        for i in range(int(np.ceil(N / batch_size))):
            sl = np.index_exp[i * batch_size:(i + 1) * batch_size]
            if nms_cont:
                res_dict['Probs_ps'][sl], res_dict['XO_ps'][sl], res_dict['YO_ps'][sl], res_dict['ZO_ps'][sl] \
                    = nms_func(res_dict['Probs'][sl], candi_thre,
                               res_dict['XO'][sl], res_dict['YO'][sl], res_dict['ZO'][sl])
            else:
                res_dict['Probs_ps'][sl] = nms_func(res_dict['Probs'][sl], candi_thre=candi_thre)

    res_dict['Samples_ps'] = np.where(res_dict['Probs_ps'] > threshold, 1, 0)  # deterministic locs




def nms_func(p, candi_thre=0.3, xo=None, yo=None, zo=None):
    with torch.no_grad():
        diag = 0  # 1/np.sqrt(2)

        p = gpu(p)

        p_copy = p + 0

        # probability values > 0.3 are regarded as possible locations

        # p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]
        p_clip = torch.where(p > candi_thre, p, torch.zeros_like(p))[:, None]  # fushuang

        # localize maximum values within a 3x3 patch

        pool = func.max_pool2d(p_clip, 3, 1, padding=1)
        max_mask1 = torch.eq(p[:, None], pool).float()

        # Add probability values from the 4 adjacent pixels

        filt = np.array([[diag, 1, diag], [1, 1, 1], [diag, 1, diag]], ndmin=4)
        conv = func.conv2d(p[:, None], gpu(filt), padding=1)
        p_ps1 = max_mask1 * conv

        # In order do be able to identify two fluorophores in adjacent pixels we look for probablity values > 0.6 that are not part of the first mask

        p_copy *= (1 - max_mask1[:, 0])
        p_clip = torch.where(p_copy > 0.6, p_copy, torch.zeros_like(p_copy))[:, None]  # fushuang
        max_mask2 = torch.where(p_copy > 0.6, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:, None]  # fushuang
        p_ps2 = max_mask2 * conv

        # This is our final clustered probablity which we then threshold (normally > 0.7) to get our final discrete locations 
        p_ps = p_ps1 + p_ps2

        if xo is None:
            return p_ps[:, 0].cpu()

        xo = gpu(xo)
        yo = gpu(yo)
        zo = gpu(zo)

        max_mask = torch.clamp(max_mask1 + max_mask2, 0, 1)

        mult_1 = max_mask1 / p_ps1
        mult_1[torch.isnan(mult_1)] = 0
        mult_2 = max_mask2 / p_ps2
        mult_2[torch.isnan(mult_2)] = 0

        # The rest is weighting the offset variables by the probabilities

        z_mid = zo * p
        z_conv1 = func.conv2d((z_mid * (1 - max_mask2[:, 0]))[:, None], gpu(filt), padding=1)
        z_conv2 = func.conv2d((z_mid * (1 - max_mask1[:, 0]))[:, None], gpu(filt), padding=1)

        zo_ps = z_conv1 * mult_1 + z_conv2 * mult_2
        zo_ps[torch.isnan(zo_ps)] = 0

        x_mid = xo * p
        x_mid_filt = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], ndmin=4)
        xm_conv1 = func.conv2d((x_mid * (1 - max_mask2[:, 0]))[:, None], gpu(x_mid_filt), padding=1)
        xm_conv2 = func.conv2d((x_mid * (1 - max_mask1[:, 0]))[:, None], gpu(x_mid_filt), padding=1)

        x_left = (xo + 1) * p
        x_left_filt = flip_filt(np.array([[diag, 0, 0], [1, 0, 0], [diag, 0, 0]], ndmin=4))
        xl_conv1 = func.conv2d((x_left * (1 - max_mask2[:, 0]))[:, None], gpu(x_left_filt), padding=1)
        xl_conv2 = func.conv2d((x_left * (1 - max_mask1[:, 0]))[:, None], gpu(x_left_filt), padding=1)

        x_right = (xo - 1) * p
        x_right_filt = flip_filt(np.array([[0, 0, diag], [0, 0, 1], [0, 0, diag]], ndmin=4))
        xr_conv1 = func.conv2d((x_right * (1 - max_mask2[:, 0]))[:, None], gpu(x_right_filt), padding=1)
        xr_conv2 = func.conv2d((x_right * (1 - max_mask1[:, 0]))[:, None], gpu(x_right_filt), padding=1)

        xo_ps = (xm_conv1 + xl_conv1 + xr_conv1) * mult_1 + (xm_conv2 + xl_conv2 + xr_conv2) * mult_2

        y_mid = yo * p
        y_mid_filt = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], ndmin=4)
        ym_conv1 = func.conv2d((y_mid * (1 - max_mask2[:, 0]))[:, None], gpu(y_mid_filt), padding=1)
        ym_conv2 = func.conv2d((y_mid * (1 - max_mask1[:, 0]))[:, None], gpu(y_mid_filt), padding=1)

        y_up = (yo + 1) * p
        y_up_filt = flip_filt(np.array([[diag, 1, diag], [0, 0, 0], [0, 0, 0]], ndmin=4))
        yu_conv1 = func.conv2d((y_up * (1 - max_mask2[:, 0]))[:, None], gpu(y_up_filt), padding=1)
        yu_conv2 = func.conv2d((y_up * (1 - max_mask1[:, 0]))[:, None], gpu(y_up_filt), padding=1)

        y_down = (yo - 1) * p
        y_down_filt = flip_filt(np.array([[0, 0, 0], [0, 0, 0], [diag, 1, diag]], ndmin=4))
        yd_conv1 = func.conv2d((y_down * (1 - max_mask2[:, 0]))[:, None], gpu(y_down_filt), padding=1)
        yd_conv2 = func.conv2d((y_down * (1 - max_mask1[:, 0]))[:, None], gpu(y_down_filt), padding=1)

        yo_ps = (ym_conv1 + yu_conv1 + yd_conv1) * mult_1 + (ym_conv2 + yu_conv2 + yd_conv2) * mult_2

        return p_ps[:, 0].cpu(), xo_ps[:, 0].cpu(), yo_ps[:, 0].cpu(), zo_ps[:, 0].cpu()


def array_to_list(infs, wobble=[0, 0], pix_nm=[100, 100], z_scale=700, int_scale=5000, drifts=None, start_img=0, start_n=0):
    """Transform the the output of the model (dictionary of outputs at imaging resolution) into a list of predictions.
    
    Parameters
    ----------
    infs: dict
        Dictionary of arrays created with decode_func
    wobble: list of floats
        When working with challenge data two constant offsets can be substracted from the x,y variables to
        account for shifts introduced in the PSF fitting.
    pix_nm: list of floats
        x, y pixel size (nano meter)
    drifts:
        If drifts is not None, add the drifts to the xyz.
    start_img: int
        When processing data in multiple batches this variable should be set to the last image count of the
        previous batch to get continuous counting
    start_n: int
        When processing data in multiple batches this variable should be set to the last localization count
        of the previous batch to get continuous counting
        
    Returns
    -------
    res_dict: pred_list
        List of localizations with columns: 'localization', 'frame', 'x', 'y', 'z', 'intensity',
        'x_sig', 'y_sig', 'z_sig' in the following order
            1. counter of the molecule 
            2. image number used to index into the number of the image in the prediction arrays
            3. x position in nm where 0 is top left corner
            4. y position in nm where 0 is top left corner
            5. z position in nm where 0 is from the reference 0 nm in height
            6. photon counts 
            7. probability afer nms
            8. x_sigma in nm 
            9. y_sigma in nm
            10. z_sigma in nm
            11. photon_counts_sigma
            12. x offset
            13. y offset
    """
    samples = infs['Samples_ps']  # determine which pixel has a molecule
    # probs = infs['Probs_ps']

    if drifts is None:
        drifts = np.zeros([len(samples), 4])

    pred_list = []
    count = 1 + start_n

    for i in range(len(samples)):
        pos = np.nonzero(infs['Samples_ps'][i])  # get the deterministic pixel position
        xo = infs['XO_ps'][i] - drifts[i, 1]
        yo = infs['YO_ps'][i] - drifts[i, 2]
        zo = infs['ZO_ps'][i] - drifts[i, 3]
        ints = infs['Int'][i]
        p_nms = infs['Probs_ps'][i]

        if 'XO_sig' in infs:
            xos = infs['XO_sig'][i]
            yos = infs['YO_sig'][i]
            zos = infs['ZO_sig'][i]
            int_sig = infs['Int_sig'][i]

        for j in range(len(pos[0])):
            pred_list.append([count, i + 1 + start_img,
                              (0.5 + pos[1][j] + xo[pos[0][j], pos[1][j]]) * pix_nm[0] + wobble[0],
                              (0.5 + pos[0][j] + yo[pos[0][j], pos[1][j]]) * pix_nm[1] + wobble[1],
                              zo[pos[0][j], pos[1][j]] * z_scale, ints[pos[0][j], pos[1][j]] * int_scale,
                              p_nms[pos[0][j], pos[1][j]]])
            if 'XO_sig' in infs:
                pred_list[-1] += [xos[pos[0][j], pos[1][j]] * pix_nm[0], yos[pos[0][j], pos[1][j]] * pix_nm[1],
                                  zos[pos[0][j], pos[1][j]] * z_scale, int_sig[pos[0][j], pos[1][j]] * int_scale]
            else:
                pred_list[-1] += [None, None, None, None]

            pred_list[-1] += [xo[pos[0][j], pos[1][j]] , yo[pos[0][j], pos[1][j]] ]

            count += 1

    return pred_list

def padding_shift(origin_areas, padded_areas, preds_list, pix_nm):
    """the sub-image cropped is larger than win_size, we need to select the predictions in win_size"""
    if origin_areas[0] == padded_areas[0]:
        x_offset = 0
    else:
        x_offset = 20
    if origin_areas[2] == padded_areas[2]:
        y_offset = 0
    else:
        y_offset = 20

    preds_shift = []
    for i in range(len(preds_list)):
        preds_list[i][2] -= x_offset * pix_nm[0]
        preds_list[i][3] -= y_offset * pix_nm[1]
        if preds_list[i][2] < 0 or preds_list[i][3] < 0 or \
                preds_list[i][2] > (origin_areas[1] - origin_areas[0] + 1) * pix_nm[0] or \
                preds_list[i][3] > (origin_areas[3] - origin_areas[2] + 1) * pix_nm[1]:
            continue
        preds_shift.append(preds_list[i])

    return preds_shift



def post_process(preds_areas, height, width, pixel_size=[100, 100], win_size=128, pad_w=0, pad_h=0):
    """transform the sub-area coordinate to whole-filed coordinate, correct the modification that made the raw images
    to have a size of multiple of 4, which resulted in offsets."""

    rows = int(np.ceil(height / win_size))
    columns = int(np.ceil(width / win_size))
    preds = []
    for i in range(rows * columns):
        field_xy = [i % columns * win_size, i % columns * win_size + win_size - 1, i // columns * win_size,
                    i // columns * win_size + win_size - 1]
        tmp = preds_areas[i]
        for j in range(len(tmp)):
            tmp[j][2] += field_xy[0] * pixel_size[0]
            tmp[j][3] += field_xy[2] * pixel_size[1]
        preds = preds + tmp

    # if modification was made to the raw images to have a size of multiple of 4, this resulted in offsets.
    preds = np.array(preds, dtype=np.float32)
    if len(preds):
        preds[:, 2] -= pad_w * pixel_size[0]
        preds[:, 3] -= pad_h * pixel_size[1]
    preds = sorted(preds.tolist(), key=operator.itemgetter(1))

    return preds


def recognition(model, eval_imgs_all, batch_size, use_tqdm, nms, pixel_nm, plot_num,
            win_size=128, padding=True, candidate_threshold=0.3,
            nms_threshold=0.3, padded_background=110, start_field_pos=[0, 0],
            wobble=[0, 0]):
    
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

        win_size: TODO will come back to this when you understand it fully, mostly the size of the image 
                  analyzed at test time, big images will be chopped to win_size
        padding (bool): If padding=True, this will cut a larger area (20 pixels) than win_size and
                        traverse with overlap to avoid error from incomplete PSFs at the margin.
        
        candidate_threshold (float): In the probability channel, only pixels > threshold will be
                        treated as candidates for local maximum searching.

        nms_threshold (float): This works as a threshold to filter P channel to get the 
                               deterministic pixels

        padded_background (float): the value to which areas outside the image is going to be filled with

        wobble: Constant (x, y) offset addedto the final molecule list
    Returns:
    -----------
        pred_list: A list of localizatiions with columns: 'localization_no', 'frame_no', 'x', 'y', 'z', 'intensity',
                   'x_sig', 'y_sig', 'z_sig'

    """
    # number of images of shape N x h x w
    N, h, w = eval_imgs_all.shape[0], eval_imgs_all.shape[1], eval_imgs_all.shape[2]

    # copy start field
    start_field = copy.deepcopy(start_field_pos)
    # enforce the image size to be a multiple of 4, pad with given background
    
    pad_h = 0
    pad_w = 0
    if (h % 4 != 0) or (w % 4 != 0):
        # here you can put the estimating background from the images function and swap out
        # later TODO, for now we use padded_background value instead
        if h % 4 != 0:
            new_h = (h //4  + 1)  * 4
            pad_h = new_h - h
            eval_imgs_all = np.pad(eval_imgs_all, [[0, 0], [pad_h, 0], [0, 0]],
                                   mode='constant', constant_values=padded_background)
            
            start_field_pos[1] -= pad_h
            h += pad_h
        
        if w % 4 != 0:
            new_w = (w // 4 + 1) * 4
            pad_w = new_w - w
            eval_imgs_all = np.pad(eval_imgs_all, [[0, 0], [0, 0], [pad_w, 0]],
                                   mode='constant', constant_values=padded_background)
            start_field[0] -= pad_w
            
            w += pad_w
    
    # count number of areas that you need to chop the full big images into
    area_rows = int(np.ceil(h / win_size))
    area_columns = int(np.ceil(w / win_size))

    images_areas = []
    origin_areas_list = []
    areas_list = []

    for i in range(area_rows):
        for j in range(area_columns):
            x_origin = j * win_size
            y_origin = i * win_size
            x_origin_end = w if x_origin + win_size > w else x_origin + win_size
            y_origin_end = h if y_origin + win_size > h else y_origin + win_size

            if padding:
                x_start = j * win_size if j * win_size - 20 < 0 else j * win_size - 20
                y_start = i * win_size if i * win_size - 20 < 0 else i * win_size - 20
                x_end = w if x_origin + win_size + 20 > w else x_origin + win_size + 20
                y_end = h if y_origin + win_size + 20 > h else y_origin + win_size + 20
            else:
                x_start = j * win_size
                y_start = i * win_size
                x_end = w if x_start + win_size > w else x_start + win_size
                y_end = h if y_start + win_size > h else y_start + win_size
            
            # grab sub imgs and set the background to be the same as training set
            # This is a hack, to supress all things that are outside which are padded
            # to most-defininity make no predictions
            sub_imgs_tmp = eval_imgs_all[:, y_start: y_end, x_start: x_end]
            # TODO set bg values to be same as training set
            #sub_imgs_tmp = sub_imgs_tmp - padded_background + model.data_generator.simulation_params['bg_values']

            images_areas.append(sub_imgs_tmp)
            areas_list.append([x_start + start_field[0], x_end - 1 + start_field[0],
                               y_start + start_field[1], y_end - 1 + start_field[1]])
            origin_areas_list.append([x_origin + start_field[0], x_origin_end -1 + start_field[0],
                                      y_origin + start_field[1], y_origin_end -1 + start_field[1]])

    if plot_num:
        if 0 <= plot_num - 1 < N:
            plot_areas = [{} for i in range(area_rows * area_columns + 1)]
            plot_areas[0]['raw_img'] = eval_imgs_all[plot_num - 1]
            plot_areas[0]['rows'] = area_rows
            plot_areas[0]['columns'] = area_columns
            plot_areas[0]['win_size'] = win_size
        else:
            plot_areas = []
    else:
        plot_areas = []

    del eval_imgs_all # delete to save memory on large volumes

    n_per_img = 0
    preds_areas = []
    preds_areas_rescale = []

    for i in range(area_rows * area_columns):
        field_xy = torch.tensor(areas_list[i])

        if use_tqdm:
            print('{}{}{}{}{}{}{}{}{}{}'.format('\nprocessing area:', i + 1, '/', area_rows * area_columns,
                                                    ', input field_xy:', cpu(field_xy), ', use_coordconv:',
                                                    model.net_params['use_coordconv'], ', retain locs in area:',
                                                    origin_areas_list[i]))
        else:
            print('{}{}{}{}{}{}{}{}{}{}'.format('\rprocessing area:',i+1, '/', area_rows * area_columns,
                                                    ', input field_xy:', cpu(field_xy), ', use_coordconv:',
                                                    model.net_params['use_coordconv'],', retain locs in area:',
                                                    origin_areas_list[i]), end='')
        
        # infer one are at time, you will batch each of the region into one batch
        # you give one region at a time # find a way to give cell_bg images later once
        # TODO give cell-bg images by area aswell.
        arr_infs = infer_imgs(model, images_areas[0], field_xy=field_xy, batch_size=batch_size,
                              z_scale=model.data_generator.psf_params['z_scale'],
                              int_scale=model.data_generator.psf_params['photon_scale'])

        # do nms so that you pick the thresholding on Proablitiy of molecule found on the image
        nms_sampling(arr_infs, threshold=nms_threshold, candi_thre=candidate_threshold,
                         batch_size=batch_size, nms=nms, nms_cont=False)

        # make predictions into a list
        preds_list = array_to_list(arr_infs, wobble=wobble, pix_nm=pixel_nm, z_scale=model.data_generator.psf_params['z_scale'],
                                   int_scale=model.data_generator.psf_params['photon_scale'])
        
        if padding:
            # droop the molecules in the overlap between sub-areas (padding),
            # shift the remaining molecules back to correct positions
            preds_list = padding_shift(origin_areas_list[i], areas_list[i], preds_list, pix_nm=pixel_nm)
        # add the results of each one to the list
        preds_areas.append(preds_list)


        # calculate the n_per_image, need to take account for the padding
        x_index = int((torch.tensor(origin_areas_list[i]) - torch.tensor(areas_list[i]))[0])
        y_index = int((torch.tensor(origin_areas_list[i]) - torch.tensor(areas_list[i]))[2])
        n_per_img += arr_infs['Probs'][:, y_index: y_index + win_size, x_index: x_index + win_size].sum(-1).sum(-1).mean()


        if plot_num:
            if 0 <= plot_num - 1 < N:
                for k in arr_infs.keys():
                    x_index = int((torch.tensor(origin_areas_list[i]) - torch.tensor(areas_list[i]))[0])
                    y_index = int((torch.tensor(origin_areas_list[i]) - torch.tensor(areas_list[i]))[2])
                    plot_areas[i + 1][k] = copy.deepcopy(arr_infs[k][plot_num - 1,
                                                         y_index: y_index + win_size,
                                                         x_index: x_index + win_size])

        del arr_infs
        del images_areas[0]
    print('')

    preds_frames = post_process(preds_areas, h, w, pixel_size=pixel_nm, win_size=win_size, pad_w=pad_w, pad_h=pad_h)

    return preds_frames, n_per_img, plot_areas
        


def assess(test_frame_nbr, test_csv, pred_inp, size_xy=[204800, 204800], tolerance=250, border=450,
                print_res=False, min_int=False, tolerance_ax=np.inf, segmented=False):
    """
    Matches localizations to ground truth positions and provides assessment metrics used in the
    SMLM2016 challenge.
    Parameters
    ----------
    test_frame_nbr:
        number of frames that be analyzed
    test_csv:
        Ground truth positions with columns: 'localization', 'frame', 'x', 'y', 'z', 'photons'
        Either list or str with locations of csv file.
    pred_inp:
        List of predicted localizations
    size_xy:
        Size of the FOV, which contains localizations need to be assessed (nano meter)
    tolerance:
        Localizations are matched when they are within a circle of the given radius.
    border:
        Localizations that are close to the edge of the recording are excluded because they often suffer from artifacts.
    print_res:
        If true prints a list of assessment metrics.
    min_int:
         If true only uses the brightest 75% of ground truth locations.
        This is the setting used in the leaderboard of the challenge.
        However this implementation does not exactly match the method used in the localization tool.
    tolerance_ax:
        Localizations are matched when they are closer than this value in z direction.
        Should be infinity for 2D recordings. 500nm is used for 3D recordings in the challenge.
    segmented:
        If true outputs localization evaluations of different regions of image.(not completed, unnecessary)
    Returns
    -------
    perf_dict, matches: dict, list
        Dictionary of perfomance metrics.
        List of all matches localizations for further evaluation in format: [x_gt, y_gt, z_gt, intensity_gt,
        x_pred, y_pred, z_pred,	intensity_pred,	nms_p, x_sig, y_sig, z_sig]
    """

    perf_dict = None
    matches = []

    test_list = []
    if isinstance(test_csv, str):
        with open(test_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if 'truth' not in row[0]:
                    test_list.append([float(r) for r in row])
    else:
        for r in test_csv:
            test_list.append([i for i in r])

    test_list = sorted(test_list, key=operator.itemgetter(1))  
    test_list = test_list[:find_frame(test_list, test_frame_nbr)] 
    print('{}{}{}{}{}'.format('\nevaluation on ', test_frame_nbr,
                              ' images, ', 'contain ground truth: ', len(test_list)), end='')

    # If true only uses the brightest 75% of ground truth locations.
    if min_int:
        min_int = np.percentile(np.array(test_list)[:, -1], 25)
    else:
        min_int = 0

    if isinstance(pred_inp, str):
        pred_list = []
        with open(pred_inp, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if 'truth' not in row[0]:
                    pred_list.append([float(r) for r in row])

    pred_list = copy.deepcopy(pred_inp)
    print('{}{}'.format(', preds:', len(pred_list)))

    if len(pred_list) == 0:
        perf_dict = {'recall': np.nan, 'precision': np.nan, 'jaccard': np.nan, 'f_score': np.nan, 'rmse_lat': np.nan,
                     'rmse_ax': np.nan,
                     'rmse_x': np.nan, 'rmse_y': np.nan, 'jor': np.nan, 'eff_lat': np.nan, 'eff_ax': np.nan,
                     'eff_3d': np.nan}
        print('original pred_list is empty!')
        return perf_dict, matches

    perf_dict, matches = limited_matching(test_list, pred_list, min_int, limited_x=[0, size_xy[0]],
                                          limited_y=[0, size_xy[1]], border=border,
                                          print_res=print_res, tolerance=tolerance, tolerance_ax=tolerance_ax)

    if segmented:
        _, _1 = limited_matching(test_list, pred_list, min_int, limited_x=[0, 12800],
                                 limited_y=[0, 12800], border=border,
                                 print_res=print_res, tolerance=tolerance, tolerance_ax=tolerance_ax)

        _, _1 = limited_matching(test_list, pred_list, min_int, limited_x=[38400, 51200],
                                 limited_y=[0, 12800], border=border,
                                 print_res=print_res, tolerance=tolerance, tolerance_ax=tolerance_ax)

        _, _1 = limited_matching(test_list, pred_list, min_int, limited_x=[12800, 25600],
                                 limited_y=[12800, 25600], border=border,
                                 print_res=print_res, tolerance=tolerance, tolerance_ax=tolerance_ax)

        _, _1 = limited_matching(test_list, pred_list, min_int, limited_x=[0, 12800],
                                 limited_y=[38400, 51200], border=border,
                                 print_res=print_res, tolerance=tolerance, tolerance_ax=tolerance_ax)

        _, _1 = limited_matching(test_list, pred_list, min_int, limited_x=[38400, 51200],
                                 limited_y=[38400, 51200], border=border,
                                 print_res=print_res, tolerance=tolerance, tolerance_ax=tolerance_ax)

    return perf_dict, matches

    


            

def limited_matching(truth_origin, pred_list_origin, min_int, limited_x=[0, 204800], limited_y=[0, 204800],
                     border=450, print_res=True, tolerance=250, tolerance_ax=np.inf):
    print('{}{}{}{}'.format('FOV: x=', limited_x, ' y=', limited_y))

    matches = []

    truth = copy.deepcopy(truth_origin)
    pred_list = copy.deepcopy(pred_list_origin)

    truth_array = np.array(truth)
    pred_array = np.array(pred_list)

    # filter prediction and gt according to limited_x;y
    t_inds = np.where(
        (truth_array[:, 2] < limited_x[0]) | (truth_array[:, 2] > limited_x[1]) |
        (truth_array[:, 3] < limited_y[0]) | (truth_array[:, 3] > limited_y[1]))
    p_inds = np.where(
        (pred_array[:, 2] < limited_x[0]) | (pred_array[:, 2] > limited_x[1]) |
        (pred_array[:, 3] < limited_y[0]) | (pred_array[:, 3] > limited_y[1]))
    for t in reversed(t_inds[0]):
        del (truth[t])
    for p in reversed(p_inds[0]):
        del (pred_list[p])

    if len(pred_list) == 0:
        perf_dict = {'recall': np.nan, 'precision': np.nan, 'jaccard': np.nan, 'f_score': np.nan, 'rmse_lat': np.nan,
                     'rmse_ax': np.nan,
                     'rmse_x': np.nan, 'rmse_y': np.nan, 'jor': np.nan, 'eff_lat': np.nan, 'eff_ax': np.nan,
                     'eff_3d': np.nan}
        print('after FOV segmentation, pred_list is empty!')
        return perf_dict, matches

    # delete molecules of ground truth/estimation in the margin area
    if border:
        test_arr = np.array(truth)
        pred_arr = np.array(pred_list)

        t_inds = np.where(
            (test_arr[:, 2] < limited_x[0] + border) | (test_arr[:, 2] > (limited_x[1] - border)) |
            (test_arr[:, 3] < limited_y[0] + border) | (test_arr[:, 3] > (limited_y[1] - border)))
        p_inds = np.where(
            (pred_arr[:, 2] < limited_x[0] + border) | (pred_arr[:, 2] > (limited_x[1] - border)) |
            (pred_arr[:, 3] < limited_y[0] + border) | (pred_arr[:, 3] > (limited_y[1] - border)))
        for t in reversed(t_inds[0]):
            del (truth[t])
        for p in reversed(p_inds[0]):
            del (pred_list[p])

    if len(pred_list) == 0:
        perf_dict = {'recall': np.nan, 'precision': np.nan, 'jaccard': np.nan, 'f_score': np.nan, 'rmse_lat': np.nan,
                     'rmse_ax': np.nan,
                     'rmse_x': np.nan, 'rmse_y': np.nan, 'jor': np.nan, 'eff_lat': np.nan, 'eff_ax': np.nan,
                     'eff_3d': np.nan}
        print('after border, pred_list is empty!')
        return perf_dict, matches

    print('{}{}{}{}{}'.format('after FOV and border segmentation,'
                              , 'truth: ', len(truth), ' ,preds: ', len(pred_list)))

    TP = 0
    FP = 0.0001
    FN = 0.0001
    MSE_lat = 0
    MSE_ax = 0
    MSE_vol = 0

    if len(pred_list):
        for i in range(1, int(truth_origin[-1][1]) + 1):  # traverse all gt frames

            tests = []  # gt in each frame
            preds = []  # prediction in each frame

            if len(truth) > 0:  # after border filtering and area segmentation, truth could be empty
                while truth[0][1] == i:
                    tests.append(truth.pop(0))  # put all gt in the tests
                    if len(truth) < 1:
                        break
            if len(pred_list) > 0:
                while pred_list[0][1] == i:
                    preds.append(pred_list.pop(0))  # put all predictions in the preds
                    if len(pred_list) < 1:
                        break

            # if preds is empty, it means no detection on the frame, all tests are FN
            if len(preds) == 0:
                FN += len(tests)
                continue  # no need to calculate metric
            # if the gt of this frame is empty, all preds on this frame are FP
            if len(tests) == 0:
                FP += len(preds)
                continue  # no need to calculate metric

            # calculate the Euclidean distance between all gt and preds, get a matrix [number of gt, number of preds]
            dist_arr = cdist(np.array(tests)[:, 2:4], np.array(preds)[:, 2:4])
            ax_arr = cdist(np.array(tests)[:, 4:5], np.array(preds)[:, 4:5])
            tot_arr = np.sqrt(dist_arr ** 2 + ax_arr ** 2)

            if tolerance_ax == np.inf:
                tot_arr = dist_arr

            match_tests = copy.deepcopy(tests)
            match_preds = copy.deepcopy(preds)

            if dist_arr.size > 0:
                while dist_arr.min() < tolerance:
                    r, c = np.where(tot_arr == tot_arr.min())  # select the positions pair with shortest distance
                    r = r[0]
                    c = c[0]
                    if ax_arr[r, c] < tolerance_ax and dist_arr[r, c] < tolerance:  # compare the distance and tolerance
                        if match_tests[r][-1] > min_int:  # photons should be larger than min_int

                            MSE_lat += dist_arr[r, c] ** 2
                            MSE_ax += ax_arr[r, c] ** 2
                            MSE_vol += dist_arr[r, c] ** 2 + ax_arr[r, c] ** 2
                            TP += 1
                            matches.append([match_tests[r][2], match_tests[r][3], match_tests[r][4], match_tests[r][5],
                                            match_preds[c][2], match_preds[c][3], match_preds[c][4], match_preds[c][5],
                                            match_preds[c][6], match_preds[c][-4], match_preds[c][-3],
                                            match_preds[c][-2]])

                        dist_arr[r, :] = np.inf
                        dist_arr[:, c] = np.inf
                        tot_arr[r, :] = np.inf
                        tot_arr[:, c] = np.inf

                        tests[r][-1] = -100  # photon cannot be negative, work as a flag
                        preds.pop()

                    dist_arr[r, c] = np.inf
                    tot_arr[r, c] = np.inf

            for i in reversed(range(len(tests))):
                if tests[i][-1] < min_int:  # delete matched gt
                    del (tests[i])

            FP += len(preds)  # all remaining preds are FP
            FN += len(tests)  # all remaining gt are FN

    else:
        print('after border and FOV segmentation, pred list is empty!')

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    jaccard = TP / (TP + FP + FN)
    rmse_lat = np.sqrt(MSE_lat / (TP + 0.00001))
    rmse_ax = np.sqrt(MSE_ax / (TP + 0.00001))
    rmse_vol = np.sqrt(MSE_vol / (TP + 0.00001))
    jor = 100 * jaccard / rmse_lat

    eff_lat = 100 - np.sqrt((100 - 100 * jaccard) ** 2 + 1 ** 2 * rmse_lat ** 2)
    eff_ax = 100 - np.sqrt((100 - 100 * jaccard) ** 2 + 0.5 ** 2 * rmse_ax ** 2)
    eff_3d = (eff_lat + eff_ax) / 2

    matches = np.array(matches)
    rmse_x = np.nan
    rmse_y = np.nan
    rmse_z = np.nan
    rmse_i = np.nan
    if len(matches):
        rmse_x = np.sqrt(((matches[:, 0] - matches[:, 4]) ** 2).mean())
        rmse_y = np.sqrt(((matches[:, 1] - matches[:, 5]) ** 2).mean())
        rmse_z = np.sqrt(((matches[:, 2] - matches[:, 6]) ** 2).mean())
        rmse_i = np.sqrt(((matches[:, 3] - matches[:, 7]) ** 2).mean())
    else:
        print('matches is empty!')

    if print_res:
        print('{}{:0.3f}'.format('Recall: ', recall))
        print('{}{:0.3f}'.format('Precision: ', precision))
        print('{}{:0.3f}'.format('Jaccard: ', 100 * jaccard))
        print('{}{:0.3f}'.format('RMSE_lat: ', rmse_lat))
        print('{}{:0.3f}'.format('RMSE_ax: ', rmse_ax))
        print('{}{:0.3f}'.format('RMSE_vol: ', rmse_vol))
        print('{}{:0.3f}'.format('Jaccard/RMSE: ', jor))
        print('{}{:0.3f}'.format('Eff_lat: ', eff_lat))
        print('{}{:0.3f}'.format('Eff_ax: ', eff_ax))
        print('{}{:0.3f}'.format('Eff_3d: ', eff_3d))
        print('FN: ' + str(np.round(FN)) + ' FP: ' + str(np.round(FP)))

    perf_dict = {'recall': recall, 'precision': precision, 'jaccard': jaccard, 'rmse_lat': rmse_lat,
                 'rmse_ax': rmse_ax, 'rmse_vol': rmse_vol, 'rmse_x': rmse_x, 'rmse_y': rmse_y,
                 'rmse_z': rmse_z, 'rmse_i': rmse_i, 'jor': jor, 'eff_lat': eff_lat, 'eff_ax': eff_ax,
                 'eff_3d': eff_3d}

    return perf_dict, matches


        
        

def spline_crlb_plot(calib_file, z_range=400.0, pixel_size_xy=[65.0, 65.0], dz=25.0,
                 img_size=129, step_size=10.0, photon_counts=10000.0, bg_photons=100.0,
                 ):
    """
    Plots the crlb values of the spline model. Calculated from spline PSF package that comes
    with decode
    
    Arguments:
    ------------
        calib_file: .mat file you get from SMAP

        z_range (float, nms): z_range over which you want to calculate the CRLBs

        pixel_size_xy (list of two floats): pixel size to set the voxel size in the cubic spline model

        dz: voxel size in z, you used for spline calibration

        img_size (int): odd number so that psf will be place at the center when 
                        calculating CRLB

        step_size (float): step value increments at which CRLB will be calculated.

        photon_counts (float): Each PSF will have the same photon counts

        bg_photons (float): bg value for PSF simulation

    """
    psf = SMAPSplineCoefficient(calib_file).init_spline(
        xextent=[-0.5, img_size-0.5], yextent=[-0.5, img_size-0.5], 
        img_shape=[img_size, img_size], device='cpu',
        roi_size=None, roi_auto_center=None
    )
    
    psf.vx_size = torch.tensor([pixel_size_xy[0], pixel_size_xy[1], dz])

    z = torch.arange(-z_range, z_range+step_size, step_size)

    n_planes = len(z)

    xyz = torch.zeros((len(z), 3))
    xyz[:, 0] = img_size // 2
    xyz[:, 1] = img_size // 2
    xyz[:, 2] = z
    phot = photon_counts * torch.ones((n_planes,))
    bg = bg_photons * torch.ones((n_planes,))

    crlb, _ = psf.crlb_sq(xyz, phot, bg)

    plt.figure(constrained_layout=True)
    plt.plot(z, crlb[:, 0], 'b', z, crlb[:, 1], 'g', z, crlb[:, 2], 'r')
    plt.legend((r'$\sqrt{CRLB_{x}}$', r'$\sqrt{CRLB_{y}}$' , r'$\sqrt{CRLB_{z}}$'))
    plt.xlim([z[0], z[-1]])
    plt.xlabel('Z (nm)')
    plt.ylabel(r'$\sqrt{CRLB}$ (nm)')
    plt.title(f"CRLB plot at photon counts={photon_counts} and background={bg_photons}")
    plt.show()




def model_RMSE_plot():
    """
    Plot the RMSE values in the localization to be able to compare against CRLB 
    obtainable from the spline model.

    Arguments:
    -----------

    """
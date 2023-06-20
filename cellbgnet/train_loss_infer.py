import torch.nn as nn
import numpy as np
import torch
import random
from skimage.io import imread
from skimage import segmentation
from cellbgnet.utils.hardware import cpu, gpu
from skimage.transform import rotate

def generate_probmap_cells(image, batch_size, train_size, density_in_cells, density_if_no_cells,
             margin_empty, augment=False):
    """
    Generate batch_size number of random crops from one image of size train_size
    
    Arguments:
    -----------
        image: a numpy array of cell mask (if binary mask is given, it will be labelled)
        
        batch_size (int): number of random crops to generate from one image
        
        train_size (int): size of the crop, it is a square image by default
        
        density_in_cells (int): pixel probability that a dot is present cell. (2 dots / 320 pixels)
        
        density_if_no_cells (int): total number of dots per frame if no cells were sampled in the FOV,
                                   typically 10 as default in parameter file
        
    Returns:
    --------
        prob_map (np.ndarray): a numpy array with appropriate probmap set 
                              for each image
        
    """
    # if bool, then binary mask is given, so label it.
    if (image.dtype == 'bool'):
        image = label(image)
    
    # generate corners for random cropping.
    height, width = image.shape
    x = np.random.randint(low=0, high=width - train_size, size=batch_size)
    y = np.random.randint(low=0, high=height - train_size, size=batch_size)
    
    prob_map = np.zeros([batch_size, train_size, train_size])
    cell_masks = np.zeros([batch_size, train_size, train_size])
    for i in range(prob_map.shape[0]):

        prob_map[i] = image[y[i]: y[i]+train_size, x[i]:x[i]+train_size] > 0
        cell_masks[i] = image[y[i]: y[i]+train_size, x[i]:x[i]+train_size]
        cell_masks[i][:int(margin_empty * train_size), :] = 0
        cell_masks[i][int((1-margin_empty) * train_size):, :] = 0
        cell_masks[i][:, :int(margin_empty * train_size)] = 0
        cell_masks[i][:, int((1-margin_empty) * train_size):] = 0
        
        prob_map[i][:int(margin_empty * train_size), :] = 0
        prob_map[i][int((1-margin_empty) * train_size):, :] = 0
        prob_map[i][:, :int(margin_empty * train_size)] = 0
        prob_map[i][:, int((1-margin_empty) * train_size):] = 0
        #prob_map[i][int(margin_empty * train_size) : int((1 - margin_empty) * train_size),
        #            int(margin_empty * train_size) : int((1 - margin_empty) * train_size)]
        if np.sum(prob_map[i]) == 0:
            #print('one empty')
            prob_map[i][int(margin_empty * train_size) : int((1 - margin_empty) * train_size),
                        int(margin_empty * train_size) : int((1 - margin_empty) * train_size)] += 1
            prob_map[i] = prob_map[i]/ prob_map[i].sum() * density_if_no_cells
        else:
            prob_map[i] = prob_map[i] * density_in_cells

    return prob_map, cell_masks

class TrainFuncs:

    def training(self, train_size, simulation_params):
        """
        Function that generates training data and drives the network training one step/batch

        Arguments:
            train_size (int): size of the image that the network sees
        
            simulation_params (dict): parameters to simulate, has all the options to do different kinds of simulation

        Returns:
            loss (float): total loss for a batch
        """
        density = simulation_params['density']
        # probability map describing probability of a spot
        if simulation_params['train_type'] != 'cells':
            prob_map = np.zeros([self.batch_size, train_size, train_size])
            # remove the the margins
            prob_map[0, int(simulation_params['margin_empty'] * train_size):
                int((1 - simulation_params['margin_empty']) * train_size),
                int(simulation_params['margin_empty'] * train_size):
                int((1 - simulation_params['margin_empty']) * train_size)] += 1
            prob_map = prob_map / prob_map.sum() * density

            # bg returned by datasimulator is just the psf and not the bg
            imgs_sim, xyzi_gt, s_mask, psf_imgs_gt, locs, field_xy = self.data_generator.simulate_data( 
                prob_map=gpu(prob_map), batch_size=self.batch_size,
                local_context=self.local_context,
                photon_filter=self.train_params['photon_filter'],
                photon_filter_threshold=self.train_params['photon_filter_threshold'],
                P_locs_cse=self.train_params['P_locs_cse'],
                iter_num=self._iter_count, train_size=train_size,
                robust_training=self.data_generator.simulation_params['robust_training'],
                cell_masks=None
            )
        else:
            # now you are training on cells.
            # read one image from the directory and load a random mask and generate probability map
            # to feed to the simulation of data.

            random_filename = random.choice(self.data_generator.cell_mask_filenames)
            random_cell_mask = imread(random_filename)
            random_cell_mask = segmentation.expand_labels(random_cell_mask, distance=1)
            #print('sampling random filename', random_filename)
            non_cell_density = self.simulation_params['non_cell_density']

            # generate probabilty map and cell mask crop from this one random cell mask
            prob_map, cell_masks_batch = generate_probmap_cells(random_cell_mask, self.batch_size, 
                                                train_size, density, non_cell_density,
                                                simulation_params['margin_empty'], simulation_params['augment']) 

            # bg returned by datasimulator is just the psf and not the bg
            imgs_sim, xyzi_gt, s_mask, psf_imgs_gt, locs, field_xy = self.data_generator.simulate_data( 
                prob_map=gpu(prob_map), batch_size=self.batch_size,
                local_context=self.local_context,
                photon_filter=self.train_params['photon_filter'],
                photon_filter_threshold=self.train_params['photon_filter_threshold'],
                P_locs_cse=self.train_params['P_locs_cse'],
                iter_num=self._iter_count, train_size=train_size,
                robust_training=self.data_generator.simulation_params['robust_training'],
                cell_masks=cell_masks_batch
            )
 

        P, xyzi_est, xyzi_sig, psf_imgs_est = self.inferring(imgs_sim, field_xy, 
                                                camera_chip_size=[self.data_generator.camera_chip_size[1], 
                                                                  self.data_generator.camera_chip_size[0]])

        # loss
        #loss_total, = self.final_loss(P, xyzi_est, xyzi_sig, xyzi_gt, s_mask, psf_imgs_est, psf_imgs_gt, locs)
        loss_total, (count_loss, loc_loss, bg_loss, P_locs_error) = self.final_loss(P, xyzi_est, xyzi_sig, xyzi_gt, s_mask, psf_imgs_est, psf_imgs_gt, locs)

        self.optimizer_infer.zero_grad()
        loss_total.backward()

        # avoid too large gradient
        torch.nn.utils.clip_grad_norm_(self.net_weights, 
                    max_norm=self.train_params['clip_gradient_max_norm'],
                    norm_type=2)
        
        # update the network and optimizer state
        self.optimizer_infer.step()
        self.scheduler_infer.step()

        self._iter_count += 1

        return loss_total.detach(), (count_loss.detach(), loc_loss.detach(), bg_loss.detach(), P_locs_error.detach())


    def look_trainingdata(self):
        simulation_params = self.data_generator.simulation_params
        train_size = simulation_params['train_size']
        density = simulation_params['density']

        if simulation_params['train_type'] != 'cells':
            prob_map = np.zeros([1, train_size, train_size])
            # remove the the margins
            prob_map[0, int(simulation_params['margin_empty'] * train_size):
                int((1 - simulation_params['margin_empty']) * train_size),
                int(simulation_params['margin_empty'] * train_size):
                int((1 - simulation_params['margin_empty']) * train_size)] += 1
            prob_map = prob_map / prob_map.sum() * density
        else:
            prob_map = None
        



class LossFuncs:

    # background MSE loss
    def eval_bg_sq_loss(self, psf_imgs_est, psf_imgs_gt):
        loss = nn.MSELoss(reduction='none')
        cost = loss(psf_imgs_est, psf_imgs_gt)
        cost = cost.sum(-1).sum(-1)
        return cost

    #  cross-entropy loss for probability of localization
    def eval_P_locs_loss(self, P, locs):
        loss_cse = -(locs * torch.log(P) + (1 - locs) * torch.log(1 - P))
        loss_cse = loss_cse.sum(-1).sum(-1)
        return loss_cse

    # dice loss

    # Count loss
    def count_loss_analytical(self, P, s_mask):
        log_prob = 0
        prob_mean = P.sum(-1).sum(-1)
        # TODO: understand this more carefully
        prob_var = (P - P **2).sum(-1).sum(-1)
        X = s_mask.sum(-1)
        log_prob += 1 / 2 * ((X - prob_mean) ** 2) / prob_var + 1 / 2 * torch.log(2 * np.pi * prob_var)
        return log_prob
    
    # localization loss
    def loc_loss_analytical(self, P, xyzi_est, xyzi_sig, xyzi_gt, s_mask):
                # each pixel is a component of Gaussian Mixture Model, with weights prob_normed
        prob_normed = P / (P.sum(-1).sum(-1)[:, None, None])

        p_inds = tuple((P + 1).nonzero().transpose(1, 0))

        xyzi_mu = xyzi_est[p_inds[0], :, p_inds[1], p_inds[2]]
        if self.using_gpu:
            xyzi_mu[:, 0] += p_inds[2].type(torch.cuda.FloatTensor)
            xyzi_mu[:, 1] += p_inds[1].type(torch.cuda.FloatTensor)
        else:
            xyzi_mu[:, 0] += p_inds[2].type(torch.FloatTensor)
            xyzi_mu[:, 1] += p_inds[1].type(torch.FloatTensor)

        xyzi_mu = xyzi_mu.reshape(self.batch_size, 1, -1, 4)
        xyzi_sig = xyzi_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(self.batch_size, 1, -1, 4)  # >=0.01
        # xyzi_lnsig2 = xyzi_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(self.batch_size, 1, -1, 4)  # >=0.01
        XYZI = xyzi_gt.reshape(self.batch_size, -1, 1, 4).repeat_interleave(self.train_size * self.train_size, 2)

        numerator = -1 / 2 * ((XYZI - xyzi_mu) ** 2)
        denominator = (xyzi_sig ** 2)  # >0
        # denominator = torch.exp(xyzi_lnsig2)
        log_p_gauss_4d = (numerator / denominator).sum(3) - 1 / 2 * (torch.log(2 * np.pi * denominator[:, :, :, 0]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 1]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 2]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 3]))

        gauss_coef = prob_normed.reshape(self.batch_size, 1, self.train_size * self.train_size)
        gauss_coef_logits = torch.log(gauss_coef)
        gauss_coef_logmax = torch.log_softmax(gauss_coef_logits, dim=2)

        gmm_log = torch.logsumexp(log_p_gauss_4d + gauss_coef_logmax, dim=2)
        # c = torch.sum(p_gauss_4d * gauss_coef,-1)
        # gmm_log = (torch.log(c) * s_mask).sum(-1)
        return (gmm_log * s_mask).sum(-1)

    def final_loss(self, P, xyzi_est, xyzi_sig, xyzi_gt, s_mask, psf_imgs_est, psf_imgs_gt, locs):
        count_loss = torch.mean(self.count_loss_analytical(P, s_mask) * s_mask.sum(-1))
        loc_loss = -torch.mean(self.loc_loss_analytical(P, xyzi_est, xyzi_sig, xyzi_gt, s_mask))
        bg_loss = torch.mean(self.eval_bg_sq_loss(psf_imgs_est, psf_imgs_gt)) if psf_imgs_est is not None else 0
        P_locs_error = torch.mean(self.eval_P_locs_loss(P, locs)) if locs is not None else torch.tensor([0]).cuda()

        loss_total = (0.2 * count_loss) + loc_loss + (80.0 * bg_loss) + (0.25 * P_locs_error)

        return loss_total, (count_loss, loc_loss, bg_loss, P_locs_error)
        #return loss_total

class InferFuncs:

    def inferring(self, X, field_xy, camera_chip_size):
        """
        Main function for inferring process

        Arguments:
        ------------
            X:  input image could be 4 or 3 dim

            field_xy: cell bg coordinates that you make for the images
        """

        img_h, img_w = X.shape[-2], X.shape[-1]

        # simple normalizatoin
        scaled_x = (X - self.net_params['offset']) / self.net_params['factor']

        if X.ndimension() == 3: # at test time
            scaled_x = scaled_x[:, None]
            fm_out = self.frame_module(scaled_x, field_xy, camera_chip_size)
            if self.local_context:
                zeros = torch.zeros_like(fm_out[:1])
                h_t0 = fm_out
                h_tm1 = torch.cat([zeros, fm_out], 0)[:-1]
                h_tp1 = torch.cat([fm_out, zeros], 0)[1:]
                fm_out = torch.cat([h_tm1, h_t0, h_tp1], 1)
        elif X.ndimension() == 4: # at train time we will have this
            fm_out = self.frame_module(scaled_x.reshape([-1, 1, img_h, img_w]), field_xy, camera_chip_size).reshape(-1, self.n_filters * self.n_inp, img_h, img_w)
        
        # layer norm

        fm_out_LN = nn.functional.layer_norm(fm_out, normalized_shape=[self.n_filters * self.n_inp, img_h, img_w])
        cm_in = fm_out_LN

        cm_out = self.context_module(cm_in, field_xy, camera_chip_size)
        outputs = self.out_module.forward(cm_out, field_xy, camera_chip_size)

        if self.sig_pred:
            xyzi_sig = torch.sigmoid(outputs['xyzi_sig']) + 0.001
        else:
            xyzi_sig = 0.2 * torch.ones_like(outputs['xyzi'])

        probs = torch.sigmoid(torch.clamp(outputs['p'], -16., 16.))

        xyzi_est = outputs['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # ph
        psf_est = torch.sigmoid(outputs['bg'])[:, 0] if self.psf_pred else None

        return probs[:, 0], xyzi_est, xyzi_sig, psf_est


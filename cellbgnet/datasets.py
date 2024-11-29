import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from pathlib import Path
import random
from skimage.io import imread
import edt
import pickle

from cellbgnet.generic.emitter import EmitterSet
from cellbgnet.simulation.psf_kernel import SMAPSplineCoefficient
from cellbgnet.simulation.perlin_noise import PerlinNoiseFactory
from cellbgnet.utils.hardware import cpu, gpu

class SMLMDataset(Dataset):
    """
    SMLM base dataset.
    """
    _pad_modes = (None, 'same')

    def __init__(self, *, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen,
                 frame_window: int, pad: str = None, return_em: bool):
        """
        Init new dataset.
        Args:
            em_proc: Emitter processing
            frame_proc: Frame processing
            bg_frame_proc: Background frame processing
            tar_gen: Target generator
            weight_gen: Weight generator
            frame_window: number of frames per sample / size of frame window
            pad: pad mode, applicable for first few, last few frames (relevant when frame window is used)
            return_em: return target emitter
        """
        super().__init__()

        self._frames = None
        self._emitter = None

        self.em_proc = em_proc
        self.frame_proc = frame_proc
        self.bg_frame_proc = bg_frame_proc
        self.tar_gen = tar_gen
        self.weight_gen = weight_gen

        self.frame_window = frame_window
        self.pad = pad
        self.return_em = return_em

        """Sanity"""
        self.sanity_check()

    def __len__(self):
        if self.pad is None:  # loosing samples at the border
            return self._frames.size(0) - self.frame_window + 1

        elif self.pad == 'same':
            return self._frames.size(0)

    def sanity_check(self):
        """
        Checks the sanity of the dataset, if fails, errors are raised.
        """

        if self.pad not in self._pad_modes:
            raise ValueError(f"Pad mode {self.pad} not available. Available pad modes are {self._pad_modes}.")

        if self.frame_window is not None and self.frame_window % 2 != 1:
            raise ValueError(f"Unsupported frame window. Frame window must be odd integered, not {self.frame_window}.")

    def _get_frames(self, frames, index):
        hw = (self.frame_window - 1) // 2  # half window without centre

        frame_ix = torch.arange(index - hw, index + hw + 1).clamp(0, len(frames) - 1)
        return frames[frame_ix]

    def _pad_index(self, index):

        if self.pad is None:
            assert index >= 0, "Negative indexing not supported."
            return index + (self.frame_window - 1) // 2

        elif self.pad == 'same':
            return index

    def _process_sample(self, frames, tar_emitter, bg_frame):

        """Process"""
        if self.frame_proc is not None:
            frames = self.frame_proc.forward(frames)

        if self.bg_frame_proc is not None:
            bg_frame = self.bg_frame_proc.forward(bg_frame)

        if self.em_proc is not None:
            tar_emitter = self.em_proc.forward(tar_emitter)

        if self.tar_gen is not None:
            target = self.tar_gen.forward(tar_emitter, bg_frame)
        else:
            target = None

        if self.weight_gen is not None:
            weight = self.weight_gen.forward(tar_emitter, target)
        else:
            weight = None

        return frames, target, weight, tar_emitter

    def _return_sample(self, frame, target, weight, emitter):

        if self.return_em:
            return frame, target, weight, emitter
        else:
            return frame, target, weight

class SMLMTrainDataset(SMLMDataset):
    """
    A SMLM dataset where new datasets is sampleable via the sample() method of the simulation instance
    The final processing on the frame, emitters and target is done online.
    """

    def __init__(self, *, simulator, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen, frame_window, pad,
                 return_em=False):
        
        
        super().__init__(em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=pad, return_em=return_em)
        self._frames = None
        self._emitter = None
        self._bg_frames = None
        self.simulator = simulator
        
        if self._frames is not None and self._frames.dim() != 3:
            raise ValueError("Frames must be 3 dimensional, i.e. N x H x W.")

        if self._emitter is not None and not isinstance(self._emitter, (list, tuple)):
            raise TypeError("Please split emitters in list of emitters by their frame index first.")

    def sanity_check(self):

        super().sanity_check()
        
        if self._emitter is not None and not isinstance(self._emitter, (list, tuple)):
            raise TypeError("EmitterSet shall be stored in list format, where each list item is one target emitter.")

    def sample(self, verbose: bool = False):
        """
        Sample new acquisition, i.e. a whole dataset.
        Args:
            verbose: print performance / verification information
        """

        def set_frame_ix(em):  # helper function
            em.frame_ix = torch.zeros_like(em.frame_ix)
            return em

        """Sample new dataset."""
        t0 = time.time()
        emitter, frames, bg_frames = self.simulator.sample()
        if verbose:
            print(f"Sampled dataset in {time.time() - t0:.2f}s. {len(emitter)} emitters on {frames.size(0)} frames.")

        """Split Emitters into list of emitters (per frame) and set frame_ix to 0."""
        emitter = emitter.split_in_frames(0, frames.size(0) - 1)
        emitter = [set_frame_ix(em) for em in emitter]

        self._emitter = emitter
        self._frames = frames.cpu()
        self._bg_frames = bg_frames.cpu()


    def __getitem__(self, ix):
        """
        Get a training sample.
        Args:
            ix (int): index
        Returns:
            frames (torch.Tensor): processed frames. C x H x W
            tar (torch.Tensor): target
            em_tar (optional): Ground truth emitters
        """

        """Pad index, get frames and emitters."""
        ix = self._pad_index(ix)

        tar_emitter = self._emitter[ix] if self._emitter is not None else None
        frames = self._get_frames(self._frames, ix)
        bg_frame = self._bg_frames[ix] if self._bg_frames is not None else None

        frames, target, weight, tar_emitter = self._process_sample(frames, tar_emitter, bg_frame)

        return self._return_sample(frames, target, weight, tar_emitter)


def map_masks(dists, map_values):
    dists_copy = np.copy(dists)
    for edt_val, value in map_values.items():
        dists_copy[dists == edt_val] = value
    return dists_copy

class DataSimulator(object):

    def __init__(self, psf_params, simulation_params, hardware_params):
        self.psf_params = psf_params
        self.psf_size = self.psf_params['psf_size']
        self.simulation_params = simulation_params
        self.hardware_params = hardware_params
        self.device = hardware_params['device_simulation']
        self.img_size = self.simulation_params['train_size']
        self.camera_chip_size = self.simulation_params['camera_chip_size']
        if self.device[:4] == 'cuda':
            self.use_gpu = True
        else:
            self.use_gpu = False

        # initialize cubic spline psf
        if not self.simulation_params['psf_intensity_normalization']:
            self.psf = SMAPSplineCoefficient(self.psf_params['calib_file']).init_spline(
                xextent=[-0.5, self.img_size-0.5],
                yextent=[-0.5, self.img_size-0.5],
                img_shape=[self.img_size, self.img_size],
                device='cpu',
                roi_size=None, roi_auto_center=None
            )
        else: # normalize each PSF to 1 before multiplying by photon counts
            self.psf = SMAPSplineCoefficient(self.psf_params['calib_file']).init_spline(
                xextent=[-0.5, self.psf_size-0.5],
                yextent=[-0.5, self.psf_size-0.5],
                img_shape=[self.psf_size, self.psf_size],
                device='cpu',
                roi_size=None, roi_auto_center=None
            )
        

        # setup data directory and sampling from mask files
        if self.simulation_params['train_type'] == 'cells':
            self.cell_masks_dir = Path(self.simulation_params['cell_masks_dir'])
            self.cell_mask_filenames = sorted(list(self.cell_masks_dir.glob('*' + self.simulation_params['cell_masks_filetype'])))
        else:
            self.cell_masks_dir = None
            self.cell_mask_filenames = None

        # read noise map either from file or a single fixed value
        if type(self.simulation_params['sig_read']) == str:
            self.RN = imread(self.simulation_params['sig_read'])
        else:
            self.RN = self.simulation_params['sig_read']
        
        # reading the edt noise map if you are training for cells
        if self.simulation_params['train_type'] == 'cells':
            with open(Path(self.simulation_params['edt_noise_map_path']), 'rb') as fp:
                self.edt_noise_map = pickle.load(fp)
        else:
            self.edt_noise_map = None

    def look_batch(self):
        imgs_sim, xyzi_gt, s_mask, psf_imgs_gt, locs = self.sampling()
        pass

    # function that takes in a centered psf dot on a small frame and
    # puts it on bigger images
    def place_centered_psfs(self, S, psfs_sim, i_vals):

        W = psfs_sim.to(self.device)
        W /= W.sum(-1).sum(-1)[:, None, None]

        i_vals = i_vals.to(self.device)
        # each psf dot now has the correct intesity normalized photon counts
        W *= i_vals

        recs = torch.zeros_like(S)
        h, w = S.shape[1], S.shape[2]

        s_inds = tuple(S.nonzero().transpose(1, 0))
        relu = nn.ReLU()

        rc_inds = S.nonzero()[:, 1:]

        row_rl = relu(rc_inds[:, 0] - self.psf_size // 2)
        col_rl = relu(rc_inds[:, 1] - self.psf_size // 2)

        row_start = relu(self.psf_size // 2 - rc_inds[:, 0])
        row_end = self.psf_size - relu(rc_inds[:, 0] + self.psf_size - self.psf_size//2 - h)

        col_start = relu(self.psf_size // 2 - rc_inds[:, 1])
        col_end = self.psf_size - relu(rc_inds[:, 1] + self.psf_size - self.psf_size//2 - w)


        for i in range(len(rc_inds)):
            w_cut = W[i, row_start[i]: row_end[i] , col_start[i]: col_end[i]]
            #print(i, w_cut.shape, row_rl[i], col_rl[i])
            recs[s_inds[0][i], row_rl[i]:row_rl[i] + w_cut.shape[0], col_rl[i]:col_rl[i] + w_cut.shape[1]] += w_cut
        return recs

    def simulate_psfs(self, S, X_os, Y_os, Z, I, robust_training=False):

        if not self.simulation_params['psf_intensity_normalization']:
            batch_size, n_inp, h, w = S.shape[0], S.shape[1], S.shape[2], S.shape[3]
            xyzi = torch.cat([X_os.reshape([-1, 1, h, w]), Y_os.reshape([-1, 1, h, w]), Z.reshape([-1, 1, h, w]),
                            I.reshape([-1, 1, h, w])], 1)
        
            S = S.reshape([-1, h, w])
            n_samples = S.shape[0] // xyzi.shape[0]
            XYZI_rep = xyzi.repeat_interleave(n_samples, 0)

            #print(XYZI_rep.shape)
            s_inds = tuple(S.nonzero().transpose(1, 0))
            #print(s_inds)
            x_os_vals = (XYZI_rep[:, 0][s_inds])[:, None, None]
            y_os_vals = (XYZI_rep[:, 1][s_inds])[:, None, None]
            # z_vals will be between -1 and 1, so they will be scaled to nm's here
            z_vals = self.psf_params['z_scale'] * (XYZI_rep[:, 2][s_inds])[:, None, None]
            i_vals = self.psf_params['photon_scale'] * (XYZI_rep[:, 3][s_inds])[:, None, None]

            n_emitters = len(s_inds[0])
            xyz = torch.zeros((n_emitters, 3))
            xyz[:, 0] = s_inds[1] + x_os_vals[:, 0, 0]
            xyz[:, 1] = s_inds[2] + y_os_vals[:, 0, 0]
            xyz[:, 2] = z_vals[:, 0, 0]
            photon_counts = i_vals[:, 0, 0]
            frame_ix = s_inds[0]

            em = EmitterSet(xyz=torch.index_select(xyz, 1, torch.LongTensor([1, 0, 2])),
                            phot=photon_counts.cpu(), frame_ix=frame_ix.long().cpu(), 
                            id=torch.arange(n_emitters).long(), xy_unit='px',
                            px_size=self.psf_params['pixel_size_xy'])
            

            imgs_sim =  self.psf.forward(em.xyz_px, em.phot, em.frame_ix, ix_low=0, ix_high=batch_size-1)

            torch.clamp_min_(imgs_sim, 0)
            #print(imgs_sim.shape)
            imgs_sim = imgs_sim.reshape([batch_size, n_inp, h, w])

            return imgs_sim.to(self.device)
        else:
            # place one psf at a time, normalize splinePSF slice to 1, then
            # multiply by appropriate photon counts
            #print('Doing intesity normalized psf placement')
            batch_size, n_inp, h, w = S.shape[0], S.shape[1], S.shape[2], S.shape[3]
            xyzi = torch.cat([X_os.reshape([-1, 1, h, w]), Y_os.reshape([-1, 1, h, w]), Z.reshape([-1, 1, h, w]),
                I.reshape([-1, 1, h, w])], 1)
            
            # extra not for our purpose as we don't model blinking dots for now
            S = S.reshape([-1, h, w]) 
            n_samples = S.shape[0] // xyzi.shape[0]
            XYZI_rep  = xyzi.repeat_interleave(n_samples, 0)

            s_inds = tuple(S.nonzero().transpose(1, 0))
            
            x_os_vals = (XYZI_rep[:, 0][s_inds])[:, None, None]
            y_os_vals = (XYZI_rep[:, 1][s_inds])[:, None, None]

            # z_vals will be between -1 and 1 , so they will have to be in nm's 
            # for the splinePSF model sampling
            z_vals = self.psf_params['z_scale'] * (XYZI_rep[:, 2][s_inds])[:, None, None]
            i_vals = self.psf_params['photon_scale'] * (XYZI_rep[:, 3][s_inds])[:, None, None]

            n_emitters = len(s_inds[0])
            xyz = torch.zeros((n_emitters, 3))
            psf_int_vals = torch.ones((n_emitters, ))

            xyz[:, 0] = self.psf_size // 2 + x_os_vals[:, 0, 0] 
            xyz[:, 1] = self.psf_size // 2 + y_os_vals[:, 0, 0]
            xyz[:, 2] = z_vals[:, 0, 0]
            # each frame will have one emitter only
            frame_ix = torch.arange(n_emitters)

            # it is fine to do on CPU as we don't do decode like large pre-sampling batches
            # it will be slow, but whatever.. 
            em = EmitterSet(xyz=torch.index_select(xyz, 1, torch.LongTensor([1, 0, 2])), phot=psf_int_vals.cpu(), frame_ix=frame_ix.long().cpu(),
                            id=torch.arange(n_emitters).long(), xy_unit='px',
                            px_size=self.psf_params['pixel_size_xy'])
            
            # not ix_high is n_emitters and not batch_size, as only one emitter is put on one frame
            psfs_sim = self.psf.forward(em.xyz_px, em.phot, em.frame_ix, ix_low=0, ix_high=n_emitters-1)

            imgs_sim = self.place_centered_psfs(S, psfs_sim, i_vals)
            torch.clamp_min_(imgs_sim, 0)
            imgs_sim = imgs_sim.reshape([batch_size, n_inp, h, w])

            return imgs_sim.to(self.device)

    def simulate_noise(self, imgs_sim, field_xy, cells_bg_alpha=None,
                       cells_bg_beta = None, add_noise=True):
                         
        if self.simulation_params['camera'] == 'sCMOS' or self.simulation_params['camera'] == 'EMCCD':
            # generate background photons from cells using sampled technique
            if (cells_bg_alpha is not None)  and (cells_bg_beta is not None):
                alpha_t = torch.from_numpy(cells_bg_alpha)
                beta_t = 1.0/torch.from_numpy(cells_bg_beta)

                bg_sampled_ADU = torch.distributions.gamma.Gamma(concentration=alpha_t, rate=beta_t).sample()
                bg_photons = ((bg_sampled_ADU - self.simulation_params['baseline']) * self.simulation_params['e_per_adu'])/ self.simulation_params['qe']
                bg_photons = bg_photons[:, None] # add an extra dimension so that bg_photons becomes [batch_size, 1, H, W]
                bg_photons = torch.clamp(bg_photons, min=0.0)
            else:
                # will sample batch_size number of values
                bg_sampled_ADU = torch.distributions.Uniform(low=(self.simulation_params['bg_values'] - self.simulation_params['bg_width']),
                                                             high=(self.simulation_params['bg_values'] + self.simulation_params['bg_width'])).sample(sample_shape=torch.Size([imgs_sim.shape[0]]))

                bg_sampled_ADU = bg_sampled_ADU[:, None, None]
                # batch_size, H, W
                bg_sampled_ADU = torch.ones((imgs_sim.shape[0], imgs_sim.shape[2], imgs_sim.shape[3])) * bg_sampled_ADU
                if self.simulation_params['camera'] == 'EMCCD' and self.simulation_params['em_gain'] is not None:
                    EMgain = self.simulation_params['em_gain']
                else:
                    EMgain = 1
                bg_photons = (((bg_sampled_ADU - self.simulation_params['baseline']) * self.simulation_params['e_per_adu']) / EMgain - self.simulation_params['spurious_c'])/ self.simulation_params['qe']
                bg_photons = bg_photons[:, None] # batchsize, 1, H, W
                bg_photons = torch.clamp(bg_photons, min=0.0)


            #if bg_photons.a<  0:
            #    print('Converted bg_photons is less than 0, please check parameters, bg_values and baseline')

            if self.simulation_params['perlin_noise']:
                size_x, size_y = imgs_sim.shape[-2], imgs_sim.shape[-1]
                self.PN_res = self.simulation_params['perlin_noise_res']
                self.PN_octaves_num = 1
                space_range_x = size_x / self.PN_res
                space_range_y = size_y / self.PN_res

                # initialize noise factory
                self.PN = PerlinNoiseFactory(dimension=2, octaves=self.PN_octaves_num,
                                        tile=(space_range_x, space_range_y),
                                        unbias=True)

                PN_tmp_map = np.zeros([size_x, size_y])
                for x in range(size_x):
                    for y in range(size_y):
                        cal_PN_tmp = self.PN(x / self.PN_res, y / self.PN_res)
                        PN_tmp_map[x, y] = cal_PN_tmp

                # you have pattern and you multiply by photons and a factors
                PN_noise = PN_tmp_map * bg_photons * self.simulation_params['perlin_noise_factor']
                # add perlin noise on top of already calculated bg_photons
                bg_photons += PN_noise
                if self.use_gpu:
                    bg_photons = gpu(bg_photons)

            # add bg_photons on top of psf simulated images
            if self.use_gpu:
                bg_photons = gpu(bg_photons)
            imgs_sim += bg_photons 

            if add_noise:
                imgs_sim = torch.distributions.Poisson(
                    imgs_sim * self.simulation_params['qe']  + self.simulation_params['spurious_c']).sample()
                if self.simulation_params['camera'] == 'sCMOS':
                    if type(self.RN) == np.ndarray:
                        RN = gpu(self.RN[field_xy[2] : field_xy[3]+1, field_xy[0]: field_xy[1] + 1] + 1e-6)
                        #print(f"Using read noise map for field: {field_xy}")
                    else:
                        RN = self.RN
                elif self.simulation_params['camera'] == 'EMCCD':
                    # EMCCD part
                    if type(self.RN) == np.ndarray:
                        raise ValueError('sig_read should be a value for EMCCD camera, not a matrix.')
                    else:
                        RN = self.RN
                    if self.simulation_params['em_gain'] is not None:
                        imgs_sim = torch.distributions.gamma.Gamma(imgs_sim, 1 / self.simulation_params['em_gain'], validate_args=False).sample()
                        
                        
                zeros = torch.zeros_like(imgs_sim)
                readout_noise = torch.distributions.Normal(zeros, zeros + RN).sample()
                # add read out noise
                imgs_sim = imgs_sim + readout_noise
                # convert off the electrons into ADU's and then 
                imgs_sim = torch.clamp((imgs_sim / self.simulation_params['e_per_adu']) + self.simulation_params['baseline'], min=0) 
        else:
            print('Wrong camera type. Only sCMOS and EMCCD is implemented!!')

        return imgs_sim
    
    def sampling(self, prob_map, batch_size=1, local_context=False, iter_num=None, train_size=128,
                    constant_photon_counts=False):
        """

        Randomly generate the molecule localizations, (discrete pixel + offsets) like decode but simpler
        code, with lot less redirections, photons

        Arguments:
        ------------
            constant_photon_counts: if true, will use max value as the value for all dots

        Returns:
        ------------

        """
        blink_p = prob_map
        blink_p = blink_p.reshape(blink_p.shape[0], 1, blink_p.shape[1], blink_p.shape[2])

        # Every pixel has a probability blink_p of having a molecule, following binonmial distribution
        locs1 = torch.distributions.Binomial(1, blink_p).sample().to(self.device)
        zeros = torch.zeros_like(locs1).to(self.device)

        # z positions are uniform distributino with a predefined range
        z = torch.distributions.Uniform(zeros + self.simulation_params['z_prior'][0],
                                        zeros + self.simulation_params['z_prior'][1]).sample().to(self.device)
        
        # xy offest follow uniform distribution
        x_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to(self.device)
        y_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to(self.device)

        if local_context:
            surv_p = self.simulation_pars['survival_prob']
            a11 = 1 - (1 - blink_p) * (1 - surv_p)
            locs2 = torch.distributions.Binomial(1, (1 - locs1) * blink_p + locs1 * a11).sample().to(self.device)
            locs3 = torch.distributions.Binomial(1, (1 - locs2) * blink_p + locs2 * a11).sample().to(self.device)
            locs = torch.cat([locs1, locs2, locs3], 1)
            x_os = x_os.repeat_interleave(3, 1)
            y_os = y_os.repeat_interleave(3, 1)
            z = z.repeat_interleave(3, 1)
        else:
            locs = locs1

        if constant_photon_counts == False:
            # photon number is sampled from a  uniform distribution
            ints = torch.distributions.Uniform(torch.zeros_like(locs) + self.simulation_params['min_photon'],
                                        torch.ones_like(locs)).sample().to(self.device)
        else:
            ints = 0.75 * torch.ones_like(locs).to(self.device)

        # only get values of the offsets where there are emitters
        x_os *= locs
        y_os *= locs
        z *= locs
        ints *= locs

        # when generating training data we only use train_size * train_size image
        if prob_map.shape[1] == prob_map.shape[2] == train_size:
            index  = iter_num % len(self.sliding_win)
            # get the fov of the train size from the full camera chip
            field_xy = torch.tensor(self.sliding_win[index])
        # When we use the full camera chip
        else:
            field_xy = torch.tensor([0, self.camera_chip_size[1] - 1, 0, self.camera_chip_size[0] - 1])
        
        #print(f"After sampling field_xy is: {field_xy}")
        
        return locs, x_os, y_os, z, ints, field_xy



    def simulate_data(self, prob_map, batch_size=1, local_context=False,
            photon_filter=False, photon_filter_threshold=100, P_locs_cse=False,
            iter_num=None, train_size=128, robust_training=False, cell_masks=None,
            constant_photon_counts=False):
        """
        Main function for simulating SMLM images. All different kinds of simulation modes
        get handled slightly differently

        Arguments:
        -----------
            prob_map (np.ndarray): A map with pixel values indicating the probability
                      of a molecule existing in that pixel

            batch_size (int):  Number of images simulated for each iteration

            local_context (bool): Generate 1(False) or 3 consecutive (True) frames, we don't use it 
                                  for anything
            photon_filter (bool): Enforce the network to ignore too dim molecules            
            photon_filter_threshold (int): Threshold to decide which molecules to be ignored.

            P_locs_cse (bool): If True, loss function will add and use cross-entropy term

            iter_num:
                Number of simulating iterations, (not used anywhere)
            
            train_size (int): Size of the images seen by the net
            robust_training (bool): Not used for now

            constant_photon_counts: if true, will use max photon count value in photon_scale parameter
                                    as photon counts for all dots

            cell_masks (np.ndarray): cell masks to be used for different noise models of the
                                     background. They are not used for PSF placement as it is 
                                     covered in the prob_map

        Returns:
        ---------
            imgs_sim: simulated image with PSFs 
            xyzi_gt:
            s_mask:
            psf_imgs_gt:
            locs:
        """
        if self.use_gpu:
            imgs_sim = torch.zeros([batch_size, 3, prob_map.shape[1], prob_map.shape[2]]).type(torch.cuda.FloatTensor) \
                if local_context else torch.zeros([batch_size, 1, prob_map.shape[1], prob_map.shape[2]]).type(torch.cuda.FloatTensor)
            xyzi_gt = torch.zeros([batch_size, 0, 4]).type(torch.cuda.FloatTensor)
            s_mask = torch.zeros([batch_size, 0]).type(torch.cuda.FloatTensor)
            pix_cor = torch.zeros([batch_size, 0, 2]).type(torch.cuda.FloatTensor)
        else:
            imgs_sim = torch.zeros([batch_size, 3, prob_map.shape[1], prob_map.shape[2]]).type(torch.FloatTensor) \
                if local_context else torch.zeros([batch_size, 1, prob_map.shape[1], prob_map.shape[2]]).type(torch.FloatTensor)
            xyzi_gt = torch.zeros([batch_size, 0, 4]).type(torch.FloatTensor)
            s_mask = torch.zeros([batch_size, 0]).type(torch.FloatTensor)
            pix_cor = torch.zeros([batch_size, 0, 2]).type(torch.FloatTensor)

        # do the sampling and construct what is needed. 
        # S is locs, X_os, Y_os are offsets, Z and I are also scaled values in appropriate ranges
        S, X_os, Y_os, Z, I, field_xy = self.sampling(batch_size=batch_size, prob_map=prob_map, local_context=local_context,
                                    iter_num=iter_num, train_size=train_size, constant_photon_counts=constant_photon_counts)

        # if there are emitters sampled on the image
        if S.sum():
            imgs_sim += self.simulate_psfs(S, X_os, Y_os, Z, I, robust_training=robust_training)
            xyzi = torch.cat([X_os[:, :, None], Y_os[:, :, None], Z[:, :, None], I[:, :, None]], 2)

            S = S[:, 1] if local_context else S[:, 0]

            if S.sum():
                # if simulate the local context, take the middle frame otherwise the first one
                xyzi = xyzi[:, 1] if local_context else xyzi[:, 0]
                # get all molecules' discrete pixel positions [number_in_batch, row, column]
                s_inds = tuple(S.nonzero().transpose(1, 0))
                # get these molecules' sub-pixel xy offsets, z positions and photons
                xyzi_true = xyzi[s_inds[0], :, s_inds[1], s_inds[2]]
                # get the xy continuous pixel positions
                if self.use_gpu:
                    xyzi_true[:, 0] += s_inds[2].type(torch.cuda.FloatTensor)
                    xyzi_true[:, 1] += s_inds[1].type(torch.cuda.FloatTensor)
                else:
                    xyzi_true[:, 0] += s_inds[2].type(torch.FloatTensor) 
                    xyzi_true[:, 1] += s_inds[1].type(torch.FloatTensor)
                # return the gt numbers of molecules on each training images of this batch
                # (if local_context, return the number of molecules on the middle frame)
                s_counts = torch.unique_consecutive(s_inds[0], return_counts=True)[1]
                s_max = s_counts.max()
                # for each training images of this batch, build a molecule list with length=s_max
                if self.use_gpu:
                    xyzi_gt_curr = torch.cuda.FloatTensor(batch_size, s_max, 4).fill_(0)
                    s_mask_curr = torch.cuda.FloatTensor(batch_size, s_max).fill_(0)
                    pix_cor_curr = torch.cuda.LongTensor(batch_size, s_max, 2).fill_(0)
                else:
                    xyzi_gt_curr = torch.FloatTensor(batch_size, s_max, 4).fill_(0)
                    s_mask_curr = torch.FloatTensor(batch_size, s_max).fill_(0)
                    pix_cor_curr = torch.LongTensor(batch_size, s_max, 2).fill_(0)
                    
                s_arr = torch.cat([torch.arange(c) for c in s_counts], dim=0)
                # put the gt in the molecule list, with remaining=0
                xyzi_gt_curr[s_inds[0], s_arr] = xyzi_true
                s_mask_curr[s_inds[0], s_arr] = 1
                pix_cor_curr[s_inds[0], s_arr, 0] = s_inds[1].clone().detach()
                pix_cor_curr[s_inds[0], s_arr, 1] = s_inds[2].clone().detach()

                xyzi_gt = torch.cat([xyzi_gt, xyzi_gt_curr], 1)
                s_mask = torch.cat([s_mask, s_mask_curr], 1)
                pix_cor = torch.cat([pix_cor, pix_cor_curr], 1)
            

        # add noise, bg is actually the normalized un-noised PSF image
        # not sure why we are multiplying by 10 TODO
        psf_imgs_gt = imgs_sim.clone() / self.psf_params['photon_scale'] * 10
        #psf_imgs_gt = imgs_sim.clone()
        psf_imgs_gt = psf_imgs_gt[:, 1] if local_context else psf_imgs_gt[:, 0]

        # 
        
        if cell_masks is not None:
            dists = np.zeros_like(cell_masks, dtype='float32')
            for i in range(len(cell_masks)):
                dists[i] = edt.edt(cell_masks[i])
                dists[i] = np.round(dists[i])
                dists[i] = np.clip(dists[i], self.simulation_params['min_edt'], self.simulation_params['max_edt'])
            cells_bg_alpha = map_masks(dists, self.edt_noise_map['alphas'])
            cells_bg_beta = map_masks(dists, self.edt_noise_map['betas'])
        else:
            cells_bg_alpha = None
            cells_bg_beta = None

        imgs_sim = self.simulate_noise(imgs_sim, field_xy, cells_bg_alpha, cells_bg_beta)

        # only return the ground truth if photon > threshold
        if photon_filter:
            for i in range(xyzi_gt.shape[0]):
                for j in range(xyzi_gt.shape[1]):
                    if xyzi_gt[i, j, 3] * self.psf_params['photon_scale'] < photon_filter_threshold:
                        xyzi_gt[i, j] = torch.tensor([0, 0, 0, 0])
                        s_mask[i, j] = 0
                        S[i, int(pix_cor[i, j][0]), int(pix_cor[i, j][1])] = 0
        locs = S if P_locs_cse else None

        return imgs_sim, xyzi_gt, s_mask, psf_imgs_gt, locs, field_xy

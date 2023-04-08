import time
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from cellbgnet.generic import emitter
from cellbgnet.simulation.psf_kernel import SMAPSplineCoefficient

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


class DataSimulator():

    def __init__(self, psf_params, simulation_params, hardware_params):
        self.psf_params = psf_params
        self.psf_size = self.psf_params['psf_size']
        self.simulation_params = simulation_params
        self.hardware_params = hardware_params
        self.device = hardware_params.device_simulation
        if self.device[:4] == 'cuda':
            self.use_gpu = True
        else:
            self.use_gpu = False

        # initialize cubic spline psf
        self.psf = SMAPSplineCoefficient() 

    def look_psfs(self):
        pass

    def place_psfs(self):
        pass

    def simulate_psfs(self, S, X_os, Y_os, Z, I, robust_training=False):
        batch_size, n_inp, h, w = S.shape[0], S.shape[1], S.shape[2], S.shape[3]


    def simulate_noise(self):
        pass
    
    def sampling(self, prob_map, batch_size=1, local_context=False, iter_num=None, train_size=128):
        """

        Randomly generate the molecule localizations, (discrete pixel + offsets) like decode but simpler
        code, with lot less redirections, photons

        Arguments:
        ------------

        Returns:
        ------------

        """
        blink_p = prob_map
        blink_p = blink_p.reshape(1, 1, blink_p.shape[-2], blink_p.shape[-1]).repeat_interleave(batch_size, 0)

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

        # photon number is sampled from a  uniform distribution
        ints = torch.distributions.Uniform(torch.zeros_like(locs) + self.simulation_params['min_photon'],
                                    torch.ones_like(locs)).sample().to(self.device)

        # only get values of the offsets where there are emitters
        x_os *= locs
        y_os *= locs
        z *= locs
        ints *= locs
        
        return locs, x_os, y_os, z, ints



    def simulate_data(self, prob_map, batch_size=1, local_context=False,
            photon_filter=False, photon_filter_threshold=100, P_locs_cse=False,
            iter_num=None, train_size=128, robust_training=False):
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

        Returns:
        ---------
            imgs_sim:
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
        S, X_os, Y_os, Z, I = self.sampling(batch_size=batch_size, prob_map=prob_map, local_context=local_context,
                                    iter_num=iter_num, train_size=train_size)

        # if there are emitters sampled on the image
        if S.sum():
            imgs_sim += self.simulate_psfs(S, X_os, Y_os, Z, I, robust_training=robust_training)

        # add noise, bg is actually the normalized un-noised PSF image

        # only return the ground truth if photon > threshold
        if photon_filter:
            pass

        locs = S if P_locs_cse else None

        return None 

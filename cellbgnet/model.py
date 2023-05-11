import collections
import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import random
from skimage.io import imread
from tqdm import tqdm

from cellbgnet.networks import UnetBGConv, OutnetBGConv
from cellbgnet.train_loss_infer import TrainFuncs, LossFuncs, InferFuncs
from cellbgnet.datasets import DataSimulator
from cellbgnet.utils.hardware import cpu, gpu
from cellbgnet.analyze_eval import recognition, assess

class CellBGModel(TrainFuncs, LossFuncs, InferFuncs):

    def __init__(self, param):
        """
        Cell-BG model

        Parameters:
        -----------
        param: a recursive namespace with all parameters
        """
        self.train_params = param.Train.to_dict()
        self.evaluation_params = param.Evaluation.to_dict()
        self.net_params = param.Network.to_dict()
        self.psf_params = param.PSF.to_dict()
        self.simulation_params = param.Simulation.to_dict()
        self.hardware_params = param.Hardware.to_dict() 

        self.device = torch.device(param.Hardware.device)
        self.using_gpu = True if param.Hardware.device[:4] == 'cuda' else False

        self.local_context = self.net_params['local_context']
        self.sig_pred = self.net_params['sig_pred']
        self.psf_pred = self.net_params['psf_pred']
        self.n_filters = self.net_params['n_filters']

        # for our case it will always be 1
        self.n_inp = 3 if self.local_context else 1
        n_features = self.n_filters * self.n_inp
        self.frame_module = UnetBGConv(n_inp=1, n_filters=self.n_filters, 
                                       n_stages=self.net_params['n_stages'],
                                       use_coordconv=self.net_params['use_coordconv']).to(self.device)
        self.context_module = UnetBGConv(n_inp=n_features, n_filters=self.n_filters,
                                         n_stages=self.net_params['n_stages'],
                                         use_coordconv=self.net_params['use_coordconv']).to(self.device) 
        self.out_module = OutnetBGConv(self.n_filters, self.sig_pred, self.psf_pred,
                                       pad=self.net_params['padding'], 
                                       ker_size=self.net_params['kernel_size'],
                                       use_coordconv=self.net_params['use_coordconv']).to(self.device)

        # Setup data simulation
        self.data_generator = DataSimulator(self.psf_params, self.simulation_params, self.hardware_params)

        self.net_weights = list(self.frame_module.parameters()) + list(self.context_module.parameters()) + list(self.out_module.parameters())
        # Setup optimizer
        self.optimizer_infer = torch.optim.AdamW(self.net_weights, lr=self.train_params['lr'],
                                                 weight_decay=self.train_params['w_decay'])
        self.scheduler_infer = torch.optim.lr_scheduler.StepLR(self.optimizer_infer, 
                                                               step_size=self.train_params['step_size'],
                                                               gamma=self.train_params['lr_decay'])

        # initialize counters
        self.recorder = {}
        self._iter_count = 0
        self.batch_size = self.train_params['batch_size']
        self.train_size = self.simulation_params['train_size']

        # file to save
        self.filename = param.InOut.filename

        # initialize sliding windows for training on one tile of size train_size typically 128x128
        self.init_sliding_window()
        # Done initialization

    def init_recorder(self):
        
        
        self.recorder['cost_hist'] = collections.OrderedDict([])
        self.recorder['update_time'] = collections.OrderedDict([])
        self.recorder['n_per_img'] = collections.OrderedDict([])
        self.recorder['recall'] = collections.OrderedDict([])
        self.recorder['precision'] = collections.OrderedDict([])
        self.recorder['jaccard'] = collections.OrderedDict([])
        self.recorder['rmse_lat'] = collections.OrderedDict([])
        self.recorder['rmse_ax'] = collections.OrderedDict([])
        self.recorder['rmse_x'] = collections.OrderedDict([])
        self.recorder['rmse_y'] = collections.OrderedDict([])
        self.recorder['rmse_vol'] = collections.OrderedDict([])
        self.recorder['jor'] = collections.OrderedDict([])
        self.recorder['eff_lat'] = collections.OrderedDict([])
        self.recorder['eff_ax'] = collections.OrderedDict([])
        self.recorder['eff_3d'] = collections.OrderedDict([])

    def init_sliding_window(self):
        """
        Initialize sliding windows, a sub area of the full camera chip size that will be used

        """
        # we use data_generator to be sure that it is initialized before you call
        # this sliding window intialization
        train_size = self.data_generator.simulation_params['train_size']
        margin_empty = self.data_generator.simulation_params['margin_empty']
        camera_chip_size = self.data_generator.camera_chip_size

        vacuum_size = int(np.ceil(train_size * margin_empty))
        overlap = 2 * vacuum_size

        row_num = int(np.ceil((camera_chip_size[0] - overlap) / (train_size - overlap)))
        column_num = int(np.ceil((camera_chip_size[1] - overlap) / (train_size - overlap)))

        sliding_win = []

        for iter_num in range(0, row_num * column_num):
            # check if the size is out of bounds of the camera ROI size
            if iter_num % column_num * (train_size - overlap) + train_size <= camera_chip_size[1]:
                x_field = iter_num % column_num * (train_size - overlap)
            else:
                x_field = camera_chip_size[1] - train_size
            
            if iter_num // column_num % row_num * (train_size - overlap) + train_size <= camera_chip_size[0]:
                y_field = iter_num // column_num % row_num * (train_size - overlap)
            else:
                y_field = camera_chip_size[0] - train_size
            
            sliding_win.append([x_field, x_field + train_size - 1, 
                                y_field, y_field + train_size - 1])

        self.data_generator.sliding_win = sliding_win

        print('training sliding windows on camera chip:')
        for i in range(0, len(sliding_win)):
            print(f"Area num: {i}, field_xy: {sliding_win[i]}")


    def init_eval_data(self):
        eval_size_x = self.evaluation_params['eval_size'][1]
        eval_size_y = self.evaluation_params['eval_size'][0]
        if self.evaluation_params['use_cell_masks'] == False:
            density = self.evaluation_params['molecules_per_img']

            # Set the probablity map 
            number_images = self.evaluation_params['number_images']
            prob_map = np.ones([number_images, eval_size_x, eval_size_y])
            # no molecules on the boundary
            prob_map[:, int(self.evaluation_params['margin_empty'] * eval_size_y): int((1 - self.evaluation_params['margin_empty']) * eval_size_y),
                        int(self.evaluation_params['margin_empty'] * eval_size_x): int((1 - self.evaluation_params['margin_empty']) * eval_size_x)] += 1
            for i in range(prob_map.shape[0]):
                prob_map[i] = (prob_map[i] / prob_map[i].sum()) * density
        else:
            # load number_images from the cell masks dir to generate probability map
            cell_masks_dir = Path(self.evaluation_params['cell_masks_dir'])
            cell_mask_filenames = sorted(list(cell_masks_dir.glob('*' + self.evaluation_params['cell_masks_filetype'])))
            random_filenames = random.choices(cell_mask_filenames, k=self.evaluation_params['number_images'])
            cell_masks_batch = []
            for filename in random_filenames:
                cell_masks_batch.append(imread(filename))
            cell_masks_batch = np.stack(cell_masks_batch)

            prob_map = (cell_masks_batch > 0) * self.evaluation_params['density_in_cells']

        ground_truth = []
        eval_imgs = np.zeros([1, eval_size_y, eval_size_x])

        for j in tqdm(range(self.evaluation_params['number_images']), desc='Eval image generation'):
            imgs_sim, xyzi_mat, s_mask, psf_est, locs = self.data_generator.simulate_data(
                    prob_map=gpu(prob_map[j][None, :]), batch_size=1, local_context=self.local_context,
                    photon_filter=False, photon_filter_threshold=0, P_locs_cse=False,
                    iter_num=self._iter_count, train_size=eval_size_x, cell_masks=cell_masks_batch[j][np.newaxis, :])
            imgs_tmp = cpu(imgs_sim)[:, 1] if self.local_context else cpu(imgs_sim)[:, 0]
            eval_imgs = np.concatenate((eval_imgs, imgs_tmp), axis = 0)
            
            # pool all the xyzi values
            for i in range(xyzi_mat.shape[1]):
                ground_truth.append(
                        [i + 1, j + 1, cpu(xyzi_mat[0, i, 0]) * self.data_generator.psf_params['pixel_size_xy'][0],
                        cpu(xyzi_mat[0, i, 1]) * self.data_generator.psf_params['pixel_size_xy'][1],
                        cpu(xyzi_mat[0, i, 2]) * self.data_generator.psf_params['z_scale'],
                        cpu(xyzi_mat[0, i, 3]) * self.data_generator.psf_params['photon_scale']]
                )

        self.evaluation_params['eval_imgs'] = eval_imgs[1:]
        self.evaluation_params['ground_truth'] = ground_truth
        self.evaluation_params['fov_size'] = [eval_size_x * self.data_generator.psf_params['pixel_size_xy'][0],
                                eval_size_y * self.data_generator.psf_params['pixel_size_xy'][1]]
        print('\neval images shape:', self.evaluation_params['eval_imgs'].shape, 'contain', len(ground_truth), 'molecules,')


        plt.figure(constrained_layout=True)
        ax_tmp = plt.subplot(1,1,1)
        img_tmp = plt.imshow(self.evaluation_params['eval_imgs'][0])
        plt.colorbar(mappable=img_tmp,ax=ax_tmp, fraction=0.046, pad=0.04)
        plt.title('the first image of eval set,check the background')
        # plt.tight_layout()
        plt.show()


    def eval_func(self, candidate_threshold=0.3, nms_threshold=0.7, print_result=False):
        if self.evaluation_params['ground_truth'] is not None:
            preds_raw, n_per_img, _ = recognition(model=self, eval_imgs_all=self.evaluation_params['eval_imgs'],
                                                  batch_size=self.evaluation_params['batch_size'], use_tqdm=False,
                                                  nms=True, candidate_threshold=candidate_threshold,
                                                  nms_threshold=nms_threshold,
                                                  pixel_nm=self.data_generator.psf_params['pixel_size_xy'],
                                                  plot_num=None,
                                                  win_size=self.data_generator.simulation_params['train_size'],
                                                  padding=True,
                                                  padded_background=self.evaluation_params['padded_background'])
            match_dict, _ = assess(test_frame_nbr=self.evaluation_params['number_images'],
                                   test_csv=self.evaluation_params['ground_truth'], pred_inp=preds_raw,
                                   size_xy=self.evaluation_params['fov_size'], tolerance=250, border=450,
                                   print_res=print_result, min_int=False, tolerance_ax=500, segmented=False)

            for k in self.recorder.keys():
                if k in match_dict:
                    self.recorder[k][self._iter_count] = match_dict[k]

            self.recorder['n_per_img'][self._iter_count] = n_per_img


    def fit(self, batch_size=16, max_iters=50000, print_output=True, print_freq=100):
        """
        Train a cellbg or decode model depending on parameters used

        Argmuments
        ----------
        batch_size (int) : Number of samples seen by the net in one iteration
        max_iters (int): Number of training iterations

        """
        self.batch_size = batch_size
        self.train_size = self.data_generator.simulation_params['train_size']
        self.print_freq = print_freq

        last_iter = self._iter_count
        total_time = 0
        best_record = -1e5
        iter_best = 0

        print('Started training .... ')

        # main loop for training
        while self._iter_count < max_iters:

            t0 = time.time()
            total_cost = []

            # Evaluate the performance and save model every print_fre iterations
            for _ in range(self.print_freq):
                loss = self.training(self.train_size, self.data_generator.simulation_params)
                total_cost.append(cpu(loss))
            total_time += (time.time()  - t0)

            self.recorder['cost_hist'][self._iter_count] = np.mean(total_cost)
            updatetime = 1000 * total_time / (self._iter_count - last_iter)
            last_iter = self._iter_count
            total_time = 0
            self.recorder['update_time'][self._iter_count] = updatetime

            if print_output:
                if self._iter_count > 1000 and self.evaluation_params['ground_truth'] is not None:
                    self.eval_func(candidate_threshold=self.evaluation_params['candidate_threshold'],
                                    nms_threshold=self.evaluation_params['nms_threshold'],
                                    print_result=self.evaluation_params['print_result'])
                    print('{}{:0.3f}'.format('JoR: ', float(self.recorder['jor'][self._iter_count])), end='')
                    # print('{}{}{:0.3f}'.format(' || ', 'Eff_lat: ', self.recorder['eff_lat'][self._iter_count]),end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Eff_3d: ', self.recorder['eff_3d'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Jaccard: ', self.recorder['jaccard'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Factor: ', self.recorder['n_per_img'][self._iter_count]),end='')
                    print('{}{}{:0.3f}'.format(' || ', 'RMSE_lat: ', self.recorder['rmse_lat'][self._iter_count]),end='')
                    print('{}{}{:0.3f}'.format(' || ', 'RMSE_ax: ', self.recorder['rmse_ax'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'RMSE_x: ', self.recorder['rmse_x'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'RMSE_y: ', self.recorder['rmse_y'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Recall: ', self.recorder['recall'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Precision: ', self.recorder['precision'][self._iter_count]),end='')
                    print('{}{}{}'.format(' || ', 'BatchNr.: ', self._iter_count), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self._iter_count]), end='')
                    print('{}{}{:0.1f}{}'.format(' || ', 'Time Upd.: ', float(updatetime), ' ms '))
                else:
                    # print('{}{:0.3f}'.format('Factor: ', self.recorder['n_per_img'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self._iter_count]), end='')
                    print('{}{}{:0.1f}{}'.format(' || ', 'Time Upd.: ', float(updatetime), ' ms '), end='')
                    print('{}{}{}'.format(' || ', 'BatchNr.: ', self._iter_count))                

            #  save the model
            if self.filename:
                if self._iter_count > 1000 and self.evaluation_params['ground_truth'] is not None:
                    best_record = self.recorder['eff_3d'][self._iter_count]
                    rmse_lat_best = self.recorder['rmse_lat'][self._iter_count]
                    rmse_ax_best = self.recorder['rmse_ax'][self._iter_count]
                    iter_best = self._iter_count
                    print('{}{:0.3f}{}{:0.3f}{}{:0.3f}{}{}'.format(
                        'saving this model, eff_3d, rmse_lat, rmse_ax and BatchNr are : ',
                        best_record, ' || ', rmse_lat_best, ' || ', rmse_ax_best, ' || ', iter_best))
                    print('\n')
                    with open(self.filename + '.pkl', 'wb') as f:
                        pickle.dump(self, f)
                else:
                    with open(self.filename + '.pkl', 'wb') as f:
                        pickle.dump(self, f)
        print('training finished!')

import collections
import pickle
import time
import torch

from cellbgnet.networks import UnetBGConv, OutnetBGConv
from cellbgnet.train_loss_infer import TrainFuncs, LossFuncs, InferFuncs
from cellbgnet.datasets import DataSimulator
from cellbgnet.utils.hardware import cpu, gpu

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
        self.data_generator = DataSimulator(self.psf_params, self.simulation_params)

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

        # file to save
        self.filename = filename
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
        self.recorder['rmse_vol'] = collections.OrderedDict([])
        self.recorder['jor'] = collections.OrderedDict([])
        self.recorder['eff_lat'] = collections.OrderedDict([])
        self.recorder['eff_ax'] = collections.OrderedDict([])
        self.recorder['eff_3d'] = collections.OrderedDict([])


    def init_eval_data(self):
        pass

    def eval_func(self):
        pass


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

            self.recorder['cost_hist'][self._iter_counter] = np.mean(total_cost)
            updatetime = 1000 * total_time / (self._iter_count - last_iter)
            last_iter = self._iter_count
            total_time = 0
            self.recorder['update_time'][self._iter_count] = updatetime

            if print_output:
                self._iter_count > 1000 and self.evaluation_params['ground_truth'] is not None:
                self.eval_func()

                print('{}{:0.3f}'.format('JoR: ', float(self.recorder['jor'][self._iter_count])), end='')
                # print('{}{}{:0.3f}'.format(' || ', 'Eff_lat: ', self.recorder['eff_lat'][self._iter_count]),end='')
                print('{}{}{:0.3f}'.format(' || ', 'Eff_3d: ', self.recorder['eff_3d'][self._iter_count]), end='')
                print('{}{}{:0.3f}'.format(' || ', 'Jaccard: ', self.recorder['jaccard'][self._iter_count]), end='')
                print('{}{}{:0.3f}'.format(' || ', 'Factor: ', self.recorder['n_per_img'][self._iter_count]),end='')
                print('{}{}{:0.3f}'.format(' || ', 'RMSE_lat: ', self.recorder['rmse_lat'][self._iter_count]),end='')
                print('{}{}{:0.3f}'.format(' || ', 'RMSE_ax: ', self.recorder['rmse_ax'][self._iter_count]), end='')
                print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self._iter_count]), end='')
                print('{}{}{:0.3f}'.format(' || ', 'Recall: ', self.recorder['recall'][self._iter_count]), end='')
                print('{}{}{:0.3f}'.format(' || ', 'Precision: ', self.recorder['precision'][self._iter_count]),end='')
                print('{}{}{}'.format(' || ', 'BatchNr.: ', self._iter_count), end='')
                print('{}{}{:0.1f}{}'.format(' || ', 'Time Upd.: ', float(updatetime), ' ms '))
            else:
                    # print('{}{:0.3f}'.format('Factor: ', self.recorder['n_per_img'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self._iter_count]), end='')
                    print('{}{}{:0.1f}{}'.format(' || ', 'Time Upd.: ', float(updatetime), ' ms '), end='')
                    print('{}{}{}'.format(' || ', 'BatchNr.: ', self._iter_count))                

            #  save the model
            if self.filename:
                if self._iter_count > 1000 and self.evaluation_pars['ground_truth'] is not None:
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

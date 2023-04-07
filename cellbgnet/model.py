import collections
import pickle
import time
import torch

from .networks import UnetBGConv, OutnetBGConv
from .train_loss_infer import TrainFuncs, LossFuncs, InferFuncs
from .datasets import DataSimulator

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
                                                
        # Done initialization

    def init_recorder(self):
        pass

    def init_eval_data(self):
        pass
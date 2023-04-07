import argparse
import os
import sys
import shutil
import socket
from pathlib import Path
import datetime
from art import tprint
import numpy as np
import matplotlib.pyplot as plt

import torch

import cellbgnet.utils

def parse_args():
    
    parser = argparse.ArgumentParser(description='Training Args')

    parser.add_argument('-i', '--device', default=None,
                        help='Specify the device string (cpu, cuda, cuda:0) and overwrite param.',
                        type=str, required=False)

    parser.add_argument('-p', '--param_file',
                        help='Specify your parameter file (.yml or .json).',
                        required=True)

    parser.add_argument('-d', '--debug', default=False, action='store_true',
                        help='Debug the specified parameter file. Will reduce ds size for example.')

    parser.add_argument('-w', '--num_worker_override',
                        help='Override the number of workers for the dataloaders.',
                        type=int)

    parser.add_argument('-n', '--no_log', default=False, action='store_true',
                        help='Set no log if you do not want to log the current run.')

    parser.add_argument('-l', '--log_folder', default='runs',
                        help='Specify the (parent) folder you want to log to. If rel-path, relative to DECODE root.')

    parser.add_argument('-c', '--log_comment', default=None,
                        help='Add a log_comment to the run.')

    args = parser.parse_args()
    return args


def train_model(param_file: str, device_overwrite: str=None, debug: bool=False,
                no_log: bool=False, num_worker_override: int=None,
                log_folder: str='runs', log_comment: str=None):
    """
    Sets up the engine to train DECODE+modifications. Includes sample simulation and the actual training.
    Args:
        param_file: parameter file path
        device_overwrite: overwrite cuda index specified by param file
        debug: activate debug mode (i.e. less samples) for fast testing
        no_log: disable logging
        num_worker_override: overwrite number of workers for dataloader
        log_folder: folder for logging (where tensorboard puts its stuff)
        log_comment: comment to the experiment

    """
    """ Load Parameters and back them up to the network output directory"""
    param_file = Path(param_file)
    param = cellbgnet.utils.param_io.ParamHandling().load_params(param_file)

    # Set some auto-scaling params --> TODO: skip for now, comeback later to fix 
    # scaling issues

    # add meta information
    param.Meta.version = cellbgnet.utils.bookkeeping.cellbgnet_state()

    """Experiment ID"""
    if not debug:
        expt_id = datetime.datetime.now().strftime(
              "%Y-%m-%d_%H-%M-%S") + '_' + socket.gethostname()
        from_ckpt = False
        if log_comment:
            expt_id = expt_id + '_' + log_comment
    else:
        expt_id = 'debug'
        from_ckpt = False

    """Set up directory for the experiment (network train run)"""
    if not from_ckpt:
        expt_path = Path(param.InOut.experiment_out) / Path(expt_id)

    if not expt_path.parent.exists():
        expt_path.parent.mkdir()
    
    if not from_ckpt:
        if debug:
            expt_path.mkdir(exist_ok=True)
        else:
            expt_path.mkdir(exist_ok=False)

    model_out = expt_path / Path('model.pt')
    ckpt_path = expt_path / Path('ckpt.pt')


    param_backup_in = expt_path / Path('param_run_in').with_suffix(param_file.suffix)
    shutil.copy(param_file, param_backup_in)

    param_backup = expt_path / Path('param_run').with_suffix(param_file.suffix)
    cellbgnet.utils.param_io.ParamHandling().write_params(param_backup, param)

    if debug:
        # TODO: here you should modify the code in paramhandling to do debug stuff
        cellbgnet.utils.param_io.ParamHandling.convert_param_debug(param)
    
    if num_worker_override is not None:
        param.Hardware.num_worker_train = num_worker_override

    """ Hardware settings """
    if device_overwrite is not None:
        device = device_overwrite
        param.Hardware.device_simulation = device_overwrite
    else:
        device = param.Hardware.device

    if torch.cuda.is_available():
        _, device_ix = cellbgnet.utils.hardware._specific_device_by_str(device)
        if device_ix is not None:
            torch.cuda.set_device(device)
    elif not torch.cuda.is_available():
        device = 'cpu'

    if sys.platform in ('linux', 'darwin'):
        os.nice(param.Hardware.unix_niceness)
    elif parma.Hardware.unix_niceness is not None:
        print(f"Cannot set niceness on platform {sys.platform}. You probably do not need to worry.")
        print(f"You are probably on windows .... ")
    
    torch.set_num_threads(param.Hardware.torch_threads)

    """Setup Log system"""

def main():
    print("Starting cellbgnet training .... ")
    tprint("CELLBGNET")

    args = parse_args()
    train_model(args.param_file, args.device, args.debug, args.no_log,
                args.num_worker_override, args.log_folder,
                args.log_comment)



if __name__ == '__main__':
    main()
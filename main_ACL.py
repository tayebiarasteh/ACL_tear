"""
Created on December 11, 2021.
main_ACL.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import pdb
import numpy as np
from torch.nn import CrossEntropyLoss
import torch
import os
import glob

from models.ACL_model import ACL_net
from config.serde import open_experiment, create_experiment, delete_experiment
from Train_Valid_ACL import Training

import warnings
warnings.filterwarnings('ignore')



def main_train_3D(global_config_path="/home/soroosh/Documents/Repositories/ACL_tear/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name'):
    """Main function for training + validation for directly 3d-wise

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/ACL_tear/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    model = ACL_net()
    loss_function = CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, num_iterations=params['num_iterations'], resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function)

    # loading the data
    train_loader = torch.ones((2, 1, 110, 281, 285))
    valid_loader = torch.ones((2, 1, 110, 281, 285))

    trainer.execute_training(train_loader=train_loader, valid_loader=valid_loader, augmentation=augment)





if __name__ == '__main__':
    main_train_3D(global_config_path="/home/soroosh/Documents/Repositories/ACL_tear/config/config.yaml",
                  valid=False, resume=False, augment=True, experiment_name='testtest')

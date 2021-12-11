"""
Created on December 11, 2021.
Train_Valid_ACL.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import os.path
import time
import pdb
from tensorboardX import SummaryWriter
import torch
import torchmetrics
import torchio as tio

from config.serde import read_config, write_config

import warnings
warnings.filterwarnings('ignore')



class Training:
    def __init__(self, cfg_path, num_iterations=100, resume=False, torch_seed=None):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        num_iterations: int
            Total number of epochs for training

        resume: bool
            if we are resuming training from a checkpoint

        torch_seed: int
            Seed used for random generators in PyTorch functions
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.num_iterations = num_iterations

        if resume == False:
            self.model_info = self.params['Network']
            self.model_info['seed'] = torch_seed or self.model_info['seed']
            self.iteration = 0
            self.best_F1 = float('inf')
            self.setup_cuda()
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path']))


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.

        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(self.model_info['seed'])
            torch.manual_seed(self.model_info['seed'])
        else:
            self.device = torch.device('cpu')


    def time_duration(self, start_time, end_time):
        """calculating the duration of training or one iteration

        Parameters
        ----------
        start_time: float
            starting time of the operation

        end_time: float
            ending time of the operation

        Returns
        -------
        elapsed_hours: int
            total hours part of the elapsed time

        elapsed_mins: int
            total minutes part of the elapsed time

        elapsed_secs: int
            total seconds part of the elapsed time
        """
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_model(self, model, optimiser, loss_function, weight=None):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function

        weight: 1D tensor of float
            class weights
        """

        # prints the network's total number of trainable parameters and
        # stores it to the experiment config
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal # of trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        self.model = model.to(self.device)
        if not weight==None:
            self.loss_weight = weight.to(self.device)
            self.loss_function = loss_function(weight=self.loss_weight)
        else:
            self.loss_function = loss_function()
        self.optimiser = optimiser

        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['num_iterations'] = self.num_iterations
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)


    def load_checkpoint(self, model, optimiser, loss_function):
        """In case of resuming training from a checkpoint,
        loads the weights for all the models, optimizers, and
        loss functions, and device, tensorboard events, number
        of iterations (epochs), and every info from checkpoint.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function
        """
        checkpoint = torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' + self.params['checkpoint_name'])
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.loss_weight = checkpoint['loss_state_dict']['weight']
        self.loss_weight = self.loss_weight.to(self.device)
        self.loss_function = loss_function(weight=self.loss_weight)
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.best_F1 = checkpoint['best_F1']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.iteration + 1)



    def execute_training(self, train_loader, valid_loader=None, augmentation=False):
        """Executes training by running training and validation at each epoch.

        Parameters
        ----------
        train_loader: Pytorch dataloader object
            training data loader

        valid_loader: Pytorch dataloader object
            validation data loader
       """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for iteration in range(self.num_iterations - self.iteration):
            self.iteration += 1
            start_time = time.time()

            train_F1, train_acc, train_loss = self.train_epoch_3D(train_loader=train_loader)
            if not valid_loader == None:
                valid_F1, valid_acc, valid_loss = self.valid_epoch_3D(valid_loader=valid_loader)

            # Validation iteration & calculate metrics
            if (self.iteration) % (self.params['epochbased_save_freq']) == 0:
                end_time = time.time()
                iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    self.calculate_tb_stats(train_F1=train_F1, train_acc=train_acc, train_loss=train_loss,
                                            valid_F1=valid_F1, valid_acc=valid_acc, valid_loss=valid_loss)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_F1, train_acc, train_loss,
                                        valid_F1, valid_acc, valid_loss)
                else:
                    self.calculate_tb_stats(train_F1=train_F1, train_acc=train_acc, train_loss=train_loss)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_F1, train_acc, train_loss)



    def train_epoch_3D(self, train_loader):
        """This is the pipeline based on Pytorch's Dataset and Dataloader

        Parameters
        ----------
        train_loader: Pytorch dataloader object
            training data loader

        Returns
        -------
        average_f1_score: float
        average training F1 score of the epoch

        average_accuracy: float
            average training accuracy of the epoch

        average_loss: float
            average training loss of the epoch
        """

        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1_score = 0.0

        # we imagine we only have one batch
        image = train_loader
        label = torch.ones((1, 1))

        label = label.long()
        image = image.float()
        image = image.to(self.device)
        label = label.to(self.device)

        self.optimiser.zero_grad()

        with torch.autograd.set_detect_anomaly(True):
            output, a_output = self.model(image)
            max_a_output = a_output.argmax(dim=2)  # get the slice with ACL

            loss = self.loss_function(output, label[:, 0])

            loss.backward()
            self.optimiser.step()

        total_loss += loss.item()

        # TODO: evaluation metric calculation

        return average_f1_score, average_accuracy, average_loss




    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs,
                       total_hours, total_mins, total_secs, train_F1, train_acc,
                       train_loss, valid_F1=None, valid_acc=None, valid_loss=None):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        Parameters
        ----------
        iteration_hours: int
            hours part of the elapsed time of each iteration

        iteration_mins: int
            minutes part of the elapsed time of each iteration

        iteration_secs: int
            seconds part of the elapsed time of each iteration

        total_hours: int
            hours part of the total elapsed time

        total_mins: int
            minutes part of the total elapsed time

        total_secs: int
            seconds part of the total elapsed time

        train_loss: float
            training loss of the model

        valid_loss: float
            validation loss of the model

        train_acc: float
            training accuracy of the model

        valid_acc: float
            validation accuracy of the model

        train_F1: float
            training F1 score of the model

        valid_F1: float
            validation F1 score of the model
        """

        # Saves information about training to config file
        self.params['Network']['num_steps'] = self.iteration
        write_config(self.params, self.cfg_path, sort_keys=True)

        # Saving the model based on the best F1
        if valid_F1:
            if valid_F1 < self.best_F1:
                self.best_F1 = valid_F1
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                           self.params['trained_model_name'])
        else:
            if train_F1 < self.best_F1:
                self.best_F1 = train_F1
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                           self.params['trained_model_name'])

        # Saving every couple of iterations
        if (self.iteration) % self.params['network_save_freq'] == 0:
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                       'iteration{}_'.format(self.iteration) + self.params['trained_model_name'])

        # Save a checkpoint every 2 iterations
        if (self.iteration) % self.params['network_checkpoint_freq'] == 0:
            torch.save({'iteration': self.iteration,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'loss_state_dict': self.loss_function.state_dict(), 'num_iterations': self.num_iterations,
                        'model_info': self.model_info, 'best_F1': self.best_F1},
                       os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' + self.params['checkpoint_name'])

        print('------------------------------------------------------'
              '----------------------------------')
        print(f'Iteration: {self.iteration}/{self.num_iterations} | '
              f'Iteration Time: {iteration_hours}h {iteration_mins}m {iteration_secs}s | '
              f'Total Time: {total_hours}h {total_mins}m {total_secs}s')
        print(f'\n\tTrain Loss: {train_loss:.4f} | Acc: {train_acc * 100:.2f}% | F1: {train_F1 * 100:.2f}%')

        if valid_loss:
            print(f'\t Val. Loss: {valid_loss:.4f} | Acc: {valid_acc * 100:.2f}% | F1: {valid_F1 * 100:.2f}%')

            # saving the training and validation stats
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'Iteration: {self.iteration}/{self.num_iterations} | Iteration Time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | Total Time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain Loss: {train_loss:.4f} | ' \
                   f'Acc: {train_acc * 100:.2f}% | ' \
                   f'F1: {train_F1 * 100:.2f}%\n\t Val. Loss: {valid_loss:.4f} | Acc: {valid_acc*100:.2f}% | F1: {valid_F1 * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'Iteration: {self.iteration}/{self.num_iterations} | Iteration Time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | Total Time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain Loss: {train_loss:.4f} | ' \
                   f'Acc: {train_acc * 100:.2f}% | F1: {train_F1 * 100:.2f}%\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def calculate_tb_stats(self, train_F1, train_acc, train_loss, valid_F1=None, valid_acc=None, valid_loss=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        Parameters
        ----------
        train_loss: float
            training loss of the model

        valid_loss: float
            validation loss of the model

        train_acc: float
            training accuracy of the model

        valid_acc: float
            validation accuracy of the model

        train_F1: float
            training F1 score of the model

        valid_F1: float
            validation F1 score of the model
        """

        self.writer.add_scalar('Train_F1', train_F1, self.iteration)
        # self.writer.add_scalar('Train_Accuracy', train_acc, self.iteration)
        self.writer.add_scalar('Train_Loss', train_loss, self.iteration)
        if valid_F1 is not None:
            self.writer.add_scalar('Valid_F1', valid_F1, self.iteration)
            # self.writer.add_scalar('Valid_Accuracy', valid_acc, self.iteration)
            self.writer.add_scalar('Valid_Loss', valid_loss, self.iteration)
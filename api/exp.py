import time
import os.path as osp
import torch

from model import CNN
from module import CNN_module
from module import (LogWandbCallback, SetupCallback, BestCheckpointCallback, EpochEndCallback)
from module import DataModule

import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from dataset import create_dataloaders




class MnistExperiment(object):
    def __init__(self, args):
        # define lightning module for CNN model
        self.method = CNN_module(CNN())
        seed_everything(42)
        # create our dataloaders
        self.data = self._get_data(args)
        # these are the directories we will create/save to
        self.save_dir = 'results'
        self.ckpt_dir = osp.join(self.save_dir, 'checkpoints') 
        # this is our wandb datalogger, make sure project name matches
        if args.mode == 'train':
            logger = WandbLogger(project=args.project_name, log_model='all')
        else:
            logger = True
        # check whether we have gpu
        accelerator = 'gpu' if torch.cuda.is_available() else 'auto'
        # create callbacks 
        callbacks = self._load_callbacks(args, logger)
        # create trainer
        self.trainer = self._init_trainer(args, callbacks, logger, accelerator)

    def _get_data(self, args):     
        train_loader, val_loader, test_loader = create_dataloaders(args.data_file, args.batch_size)
        return DataModule(train_loader, val_loader, test_loader)
        

    def _load_callbacks(self, args, logger):
        setup_callback = SetupCallback(
            mode = args.mode,
            timedate = time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            save_dir = self.save_dir,
            ckpt_dir = self.ckpt_dir
        )
        checkpoint_callback = BestCheckpointCallback(
            monitor = args.best_metric,
            filename = 'best-{epoch:02d}-{val_acc:.3f}',
            mode = 'max' if args.best_metric=='val_acc' else 'min',
            dirpath = self.ckpt_dir
        )
        epochend_callback = EpochEndCallback()
        wandb_callback = LogWandbCallback(logger)
        callbacks = [setup_callback, checkpoint_callback, epochend_callback, wandb_callback]
        #callbacks.append(LearningRateMonitor(logging_interval=None))  # to monitor scheduled learning rate
        return callbacks


    def _init_trainer(self, args, callbacks, logger, accelerator):
        trainer = Trainer(
            logger=logger,
            max_epochs=args.num_epochs,
            accelerator=accelerator,
            callbacks=callbacks
        )
        return trainer

    def train(self):
        self.trainer.fit(self.method, self.data)

    def test(self):
        ckpt = torch.load(osp.join(self.ckpt_dir, 'best.ckpt'))
        self.method.load_state_dict(ckpt['state_dict'])
        self.trainer.test(self.method, self.data)

    

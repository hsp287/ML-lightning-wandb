from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import logging
import os.path as osp
import os
import shutil

class LogWandbCallback(Callback):
    def __init__(self, wandb_logger):
        super().__init__()
        self.wandb_logger = wandb_logger
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            self.wandb_logger.log_image(key='sample_images', images=images, caption=captions)


class SetupCallback(Callback):
    def __init__(self, save_dir, ckpt_dir, mode, timedate):
        super().__init__()
        self.save_dir = save_dir
        self.ckpt_dir = ckpt_dir
        self.mode = mode
        self.timedate = timedate

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            if not osp.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if not osp.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)

            # log info to .log file
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(
                level=logging.INFO,
                filename=osp.join(self.save_dir, '{}_{}.log'.format(self.mode, self.timedate)),
                filemode='a', 
                format='%(asctime)s - %(message)s'
            )


class BestCheckpointCallback(ModelCheckpoint):
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))


class EpochEndCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        self.avg_train_loss = trainer.callback_metrics.get('train_loss')

    def on_validation_epoch_end(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        avg_val_loss = trainer.callback_metrics.get('val_loss')

        if hasattr(self, 'avg_train_loss'):
            logging.info(f"Epoch {trainer.current_epoch}: Lr: {lr:.7f} | Train Loss: {self.avg_train_loss:.7f} | Vali Loss: {avg_val_loss:.7f}")

            
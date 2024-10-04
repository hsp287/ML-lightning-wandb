import torch
import pytorch_lightning as L
from model import CNN
import torch.nn as nn
from torchmetrics.functional import accuracy
import numpy as np
import os.path as osp
import os


class CNN_module(L.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.example_input_array = torch.Tensor(64, 1, 28, 28)
        self.save_hyperparameters(ignore=['model'])
        self.test_results = []

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._compute_metrics(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss, acc = self._compute_metrics(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return pred

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=10)
        result = {'inputs': x.cpu().numpy(), 'preds': preds.cpu().numpy(), 'trues': y.cpu().numpy()}
        self.test_results.append(result)
        return result

    def on_test_epoch_end(self):
        results = {}
        for i in self.test_results[0].keys():
            results[i] = np.concatenate([batch[i] for batch in self.test_results], axis=0)

        if self.trainer.is_global_zero:
            folder_path = osp.join('results', 'saved')
            if not osp.exists(folder_path):
                os.makedirs(folder_path)

            for data in ['inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, data + '.npy'), results[data])
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        x = self.model(x)
        return x

    def _compute_metrics(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=10)
        return preds, loss, acc
        
        
        
        
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, train_loader, valid_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader
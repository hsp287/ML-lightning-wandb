from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
import torch.nn.functional as F

class mnistDataset(Dataset):
    def __init__(self, data_file, mode='train'):
        self.data = h5py.File(data_file, 'r')
        self.mode = mode
        self.img_key = 'train_images' if self.mode=='train' else 'test_images'
        self.label_key = 'train_labels' if self.mode=='train' else 'test_labels'

    def __len__(self):
        if self.mode == 'train':
            return 60000
        else:
            return 10000

    def __getitem__(self, idx):
        hf = self.data
        image = hf[self.img_key][idx]
        image = np.expand_dims(image, axis=0)   # add channel
        image = torch.tensor(image, dtype=torch.float32)

        # convert label to one-hot-vector
        label = torch.tensor(hf[self.label_key][idx], dtype=torch.long)

        return image, label
    
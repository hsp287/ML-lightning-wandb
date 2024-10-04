from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from .mnist_dataset import mnistDataset
import matplotlib.pyplot as plt


def create_dataloaders(data_file, batch_size=64):
    num_samples = 60000
    split = 0.1
    indices = list(range(num_samples))
    split = int(np.floor(split*num_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataset = mnistDataset(data_file, mode='train')
    test_dataset = mnistDataset(data_file, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def show_image_grid(inputs, labels, grid_size=(8, 8)):
    inputs = inputs.detach().cpu().numpy()
    inputs = inputs.reshape(64, 28, 28)
    
    labels = labels.detach().cpu().numpy()

    _, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10,10))
    idx = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if idx < inputs.shape[0]:
                axes[i, j].imshow(inputs[idx], cmap='gray', vmin=0, vmax=1)
                axes[i, j].axis('off')
                axes[i, j].set_title(f"No.{labels[idx]}")
            idx += 1
    plt.tight_layout()
    plt.show()



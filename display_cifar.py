import argparse
import os
from data import *
import torch
from utils.utils import sample_image, load_model
import time
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--train_on_val", type=int, default=0, help="train on val set, useful for debugging")
parser.add_argument("--dataset", type=str, default="cifar10", choices=["imagenet32", "cifar10", "cifardog", "cifarcat", "cifarcatdog"])

parser.add_argument("--n_row", type=int, default=4, help="size of sample image grid")


def sample_image(n_row, dataloader):
    """Saves a grid of generated imagenet pictures with captions"""
    captions = []
    gen_imgs = []
    # get sample captions
    done = False
    while not done:
        for (imgs, labels_batch, captions_batch) in dataloader:
            captions += captions_batch
            gen_imgs.append((imgs + 1) / 2)

            if len(captions) >= n_row ** 2:
                done = True
                break

    gen_imgs = torch.cat(gen_imgs).numpy()
    gen_imgs = np.clip(gen_imgs, 0, 1)

    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.2)

    for i in range(n_row ** 2):
        grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
        grid[i].set_title(captions[i])
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)

    save_file = f"images/{opt.dataset}.png"
    plt.savefig(save_file)
    print("saved  {}".format(save_file))
    plt.close()
    
    
if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)

    print("loading dataset")
    if opt.dataset == "imagenet32":
        train_dataset = Imagenet32Dataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = Imagenet32Dataset(train=0, max_size=1 if opt.debug else -1)
    elif opt.dataset == "cifar10":
        train_dataset = CIFAR10Dataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = CIFAR10Dataset(train=0, max_size=1 if opt.debug else -1)
    elif opt.dataset == "cifardog":
        train_dataset = CIFARDogDataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = CIFARDogDataset(train=0, max_size=1 if opt.debug else -1)
    elif opt.dataset == "cifarcat":
        train_dataset = CIFARCatDataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = CIFARCatDataset(train=0, max_size=1 if opt.debug else -1)
    elif opt.dataset == "cifarcatdog":
        train_dataset = CIFARCatDogDataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = CIFARCatDogDataset(train=0, max_size=1 if opt.debug else -1)
    else:
        raise Exception('Unknown dataset')

    print("creating dataloaders")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )
#     val_dataloader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=opt.batch_size,
#         shuffle=True,
#     )

#     print("Len train : {}, val : {}".format(len(train_dataloader), len(val_dataloader)))
    
    sample_image(opt.n_row, train_dataloader)
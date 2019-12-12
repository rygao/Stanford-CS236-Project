import numpy as np
import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import *
from tqdm import tqdm


class CaptionedImageDataset(Dataset):
    def __init__(self, image_shape, n_classes):
        self.image_shape = image_shape
        self.n_classes = n_classes

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, list):
        '''
        :param index: index of the element to be fetched
        :return: (image : torch.tensor , class_ids : torch.tensor ,captions : list(str))
        '''
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Imagenet32Dataset(CaptionedImageDataset):
    def __init__(self, root="datasets/ImageNet32", train=True, max_size=-1):
        '''
        :param dirname: str, root dir where the dataset is downloaded
        :param train: bool, true if train set else val
        :param max_size: int, truncate size of the dataset, useful for debugging
        '''
        super().__init__((3, 32, 32), 1000)
        self.root = root

        if train:
            self.dirname = os.path.join(root, "train")
        else:
            self.dirname = os.path.join(root, "val")

        self.classId2className = load_vocab_imagenet(os.path.join(root, "map_clsloc.txt"))
        data_files = sorted(os.listdir(self.dirname))
        self.images = []
        self.labelIds = []

        for i, f in enumerate(data_files):
            print("loading data file {}/{}, {}".format(i + 1, len(data_files), os.path.join(self.dirname, f)))
            data = np.load(os.path.join(self.dirname, f))
            self.images.append(data['data'])
            self.labelIds.append(data['labels'] - 1)
        self.images = np.concatenate(self.images, axis=0)
        self.labelIds = np.concatenate(self.labelIds)
        self.labelNames = [self.classId2className[y] for y in self.labelIds]

        if max_size >= 0:
            # limit the size of the dataset
            self.labelNames = self.labelNames[:max_size]
            self.labelIds = self.labelIds[:max_size]

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, list):
        image = torch.tensor(self.images[index]).reshape(3, 32, 32).float() / 128 - 1
        label = self.labelIds[index]
        caption = self.labelNames[index].replace("_", " ")
        return (image, label, caption)

    def __len__(self):
        return len(self.labelNames)


class CIFAR10Dataset(CaptionedImageDataset):
    def __init__(self, root='datasets/CIFAR10', train=True, max_size=-1):
        super().__init__((3, 32, 32), 10)
        self.dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=ToTensor())
        self.max_size = max_size if max_size > 0 else len(self.dataset)
        self.text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return self.max_size

    def __getitem__(self, item):
        img, label = self.dataset[item]
        img = 2 * img - 1
        text_label = self.text_labels[label]
        return img, label, text_label
    
    
class CIFARDogDataset(CaptionedImageDataset):
    def __init__(self, root='datasets/CIFAR10', train=True, max_size=-1):
        super().__init__((3, 32, 32), 10)
        self.dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=ToTensor())
        self.idx = np.nonzero([label==5 for _, label in self.dataset])[0]
        self.max_size = max_size if max_size > 0 else len(self.idx)
        self.text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return self.max_size

    def __getitem__(self, item):
        img, label = self.dataset[self.idx[item]]
        img = 2 * img - 1
        text_label = self.text_labels[label]
        return img, label, text_label
    
class CIFARCatDataset(CaptionedImageDataset):
    def __init__(self, root='datasets/CIFAR10', train=True, max_size=-1):
        super().__init__((3, 32, 32), 10)
        self.dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=ToTensor())
        self.idx = np.nonzero([label==3 for _, label in self.dataset])[0]
        self.max_size = max_size if max_size > 0 else len(self.idx)
        self.text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return self.max_size

    def __getitem__(self, item):
        img, label = self.dataset[self.idx[item]]
        img = 2 * img - 1
        text_label = self.text_labels[label]
        return img, label, text_label
    
class CIFARCatDogDataset(CaptionedImageDataset):
    def __init__(self, root='datasets/CIFAR10', train=True, max_size=-1):
        super().__init__((3, 32, 32), 10)
        self.dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=ToTensor())
        self.idx = np.nonzero([label in {3,5} for _, label in self.dataset])[0]
        self.max_size = max_size if max_size > 0 else len(self.idx)
        self.text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return self.max_size

    def __getitem__(self, item):
        img, label = self.dataset[self.idx[item]]
        img = 2 * img - 1
        text_label = self.text_labels[label]
        return img, label, text_label


def load_vocab_imagenet(vocab_file):
    vocab = {}
    with open(vocab_file) as f:
        for l in f.readlines():
            _, id, name = l[:-1].split(" ")
            vocab[int(id) - 1] = name.replace("_", " ")
    return vocab


if __name__ == "__main__":
    from mpl_toolkits.axes_grid1 import ImageGrid
    
    print("Testing CIFAR dataloader")
    n_row = 4
    d = CIFAR10Dataset(train=False)
    
#     fig = plt.figure(figsize=(8,8))
#     grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.25)        
#     for i, j in enumerate([2729,  802, 5611, 3829, 8274, 4453, 6138, 3206, 8800, 9092, 8960, 917, 9673,  391, 1921, 4504]):
#         img, class_label, text = d[j]
#         img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
#         img = (img+1)/2
        
#         grid[i].imshow(img)
#         grid[i].set_title(text)
#         grid[i].set_xticks([])
#         grid[i].set_yticks([])
        
#     plt.savefig(f'images/CIFAR10_most_confused.png')
#     plt.show()
    
#     fig = plt.figure(figsize=(8,8))
#     grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.25)      
#     for i, j in enumerate([9495, 3650, 2968, 9590,  439, 9108, 4234, 5264, 3845, 6738, 3333, 1875,   44,  808, 1651, 8356]):
#         img, class_label, text = d[j]
#         img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
#         img = (img+1)/2
        
#         grid[i].imshow(img)
#         grid[i].set_title(text)
#         grid[i].set_xticks([])
#         grid[i].set_yticks([])
        
#     plt.savefig(f'images/CIFAR10_least_confused.png')
#     plt.show()
    
    fig = plt.figure(figsize=(20,20))
    grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0.1)      
    for i in tqdm(range(10**2)):
        class_label = -1
        while class_label != i % 10:
            idx = np.random.randint(len(d))
            img, class_label, text = d[idx]
        img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
        img = (img+1)/2
        
        grid[i].imshow(img)
        if i < 10:
            grid[i].set_title(text)
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        
    plt.savefig(f'images/cifar10.png')
    plt.show()
    
#     fig = plt.figure(figsize=(20,4))
#     grid = ImageGrid(fig, 111, nrows_ncols=(2, 10), axes_pad=0.25) 
#     conf_labels = ['truck', 'cat', 'airplane', 'deer', 'airplane', 'deer', 'deer', 'airplane', 'deer', 'deer']
#     for i, j in enumerate([9495, 439, 9108, 63, 5252, 6344, 5264, 808, 4497, 6672,
#                            4453, 7166, 391, 6138, 802, 2729, 8274, 9850, 3829, 9673]):
#         img, class_label, text = d[j]
#         img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
#         img = (img+1)/2
        
#         grid[i].imshow(img)
#         grid[i].set_title(text if i < 10 else f'{conf_labels[i-10]}')
#         grid[i].set_xticks([])
#         grid[i].set_yticks([])
#     grid[0].set_ylabel('Least Confused,\ncorrectly\nlabeled as...')
#     grid[10].set_ylabel('Most Confused,\nmislabeled as...')
        
#     plt.savefig(f'images/CIFAR10_confused.png')
#     plt.show()




        
#     print("Testing CIFAR dataloader")
#     d = CIFAR10Dataset()
#     for i in range(2):
#         i = np.random.randint(0, len(d))
#         img, class_label, text = d[i]
#         img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
#         plt.figure(figsize=(1.5, 1.5))
#         plt.imshow((img+1)/2)
#         plt.title(text)
#         plt.savefig(f'images/CIFAR10_{i}.png')
#         plt.show()
        
#     print("Testing CIFAR Dog dataloader")
#     d = CIFARDogDataset()
#     for i in range(2):
#         i = np.random.randint(0, len(d))
#         img, class_label, text = d[i]
#         img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
#         plt.figure(figsize=(1.5, 1.5))
#         plt.imshow((img+1)/2)
#         plt.title(text)
#         plt.savefig(f'images/CIFARDog_{i}.png')
#         plt.show()
        
#     print("Testing CIFAR Cat dataloader")
#     d = CIFARCatDataset()
#     for i in range(2):
#         i = np.random.randint(0, len(d))
#         img, class_label, text = d[i]
#         img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
#         plt.figure(figsize=(1.5, 1.5))
#         plt.imshow((img+1)/2)
#         plt.title(text)
#         plt.savefig(f'images/CIFARCat_{i}.png')
#         plt.show()
        
#     print("Testing CIFAR CatDog dataloader")
#     d = CIFARCatDogDataset()
#     for i in range(2):
#         i = np.random.randint(0, len(d))
#         img, class_label, text = d[i]
#         img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
#         plt.figure(figsize=(1.5, 1.5))
#         plt.imshow((img+1)/2)
#         plt.title(text)
#         plt.savefig(f'images/CIFARCatDog_{i}.png')
#         plt.show()

#     print("Testing Imagenet32 dataloader")
#     d = Imagenet32Dataset()
#     for i in range(2):
#         i = np.random.randint(0, len(d))
#         img, class_label, text = d[i]
#         img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
#         plt.figure(figsize=(1.5, 1.5))
#         plt.imshow((img+1)/2)
#         plt.title(text)
#         plt.savefig(f'images/Imagenet32_{i}.png')
#         plt.show()

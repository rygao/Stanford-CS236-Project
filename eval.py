import argparse
import os
from data import *
from models.embedders import *
import torch
from models.pixelcnnpp import ConditionalPixelCNNpp
from utils.utils import sample_image, load_model
import time
from tqdm import tqdm
import numpy as np
import pickle as pkl

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_resnet", type=int, default=5, help="number of layers for the pixelcnn model")
parser.add_argument("--n_filters", type=int, default=160, help="dimensionality of the latent space")
parser.add_argument("--use_cuda", type=int, default=1, help="use cuda if available")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--train", type=int, default=0, help="1 = training set, 0 = validation set")
parser.add_argument("--train_on_val", type=int, default=0, help="train on val set, useful for debugging")
parser.add_argument("--model_checkpoint_folder", type=str, default=None,
                    help="load all models from checkpoint folder")
parser.add_argument("--model_checkpoint", type=str, default=None,
                    help="load model from checkpoint, model_checkpoint = path_to_your_pixel_cnn_model.pt")
parser.add_argument("--dataset", type=str, default="cifar10", choices=["imagenet32", "cifar10", "cifardog", "cifarcat", "cifarcatdog"])
parser.add_argument("--conditioning", type=str, default="unconditional", choices=["unconditional", "one-hot", "bert"])

parser.add_argument("--fixed_class_id", type=int, default=-1, help="Class label index (0-9 for CIFAR10 dataset)")
parser.add_argument("--bpds_output_file", type=str, default=None, help="Output location of bpds pickle file")


def eval(model, embedder, train=0, save_true_labels=1):
    print(f"EVALUATING ON {'TRAIN' if train else 'VAL'}")
    test_loader = train_dataloader if train else val_dataloader
    model = model.eval()
    bpd = 0.0
    bpds = []
    all_labels = []
    for i, (imgs, labels, captions) in tqdm(enumerate(test_loader), total=len(test_loader)):
        all_labels.extend(labels.cpu().numpy())
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            condition_embd = embedder(labels, captions)
            outputs = model.forward(imgs, condition_embd)
            loss = outputs['loss']
            bpds.extend(outputs['loss_by_sample'].cpu().numpy() / np.log(2))
            bpd += loss / np.log(2)
            
    bpd /= len(test_loader)
    print(f"{'TRAIN' if train else 'VAL'} bpd : {bpd}")
    
    if opt.bpds_output_file is not None:
        with open(opt.bpds_output_file, 'bw') as f:
            pkl.dump(bpds, f)
        if save_true_labels:
            with open('./true_labels.pkl', 'bw') as f:
                pkl.dump(all_labels, f)
        
    return bpd


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
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
    )
    print("dataloaders loaded")

    print("Len train : {}, val : {}".format(len(train_dataloader), len(val_dataloader)))

    device = torch.device("cuda") if (torch.cuda.is_available() and opt.use_cuda) else torch.device("cpu")
    print("Device is {}".format(device))

    print("Loading models on device...")

    # Initialize embedder
    if opt.conditioning == 'unconditional':
        encoder = UnconditionalClassEmbedding()
    elif opt.conditioning == "bert":
        encoder = BERTEncoder()
    else:
        assert opt.conditioning == "one-hot"
        encoder = OneHotClassEmbedding(train_dataset.n_classes)
        
    ## FAKE ENCODER
    if opt.fixed_class_id >= 0:
        encoder = OneHotClassEmbeddingFixed(train_dataset.n_classes, opt.fixed_class_id)

    generative_model = ConditionalPixelCNNpp(embd_size=encoder.embed_size, img_shape=train_dataset.image_shape,
                                             nr_resnet=opt.n_resnet, nr_filters=opt.n_filters,
                                             nr_logistic_mix=3 if train_dataset.image_shape[0] == 1 else 10)

    generative_model = generative_model.to(device)
    encoder = encoder.to(device)
    print("Models loaded on device")
    
    # ----------
    #  Training
    # ----------
    if opt.model_checkpoint is not None:
        model_checkpoints = [opt.model_checkpoint]
    else:
        model_checkpoints = sorted([os.path.join(opt.model_checkpoint_folder, chkpt) for chkpt in os.listdir(opt.model_checkpoint_folder)])
        
    for model_checkpoint in model_checkpoints:
        print(model_checkpoint)
        assert model_checkpoint is not None, 'no model checkpoint specified'
        print("Loading model from state dict...")
        load_model(model_checkpoint, generative_model, device)
        print("Model loaded.")
        eval(model=generative_model, embedder=encoder, train=0)
#         eval(model=generative_model, embedder=encoder, train=1)
    
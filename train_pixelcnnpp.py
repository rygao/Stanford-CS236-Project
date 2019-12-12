import argparse
import os
from data import CIFAR10Dataset, Imagenet32Dataset
from models.embedders import BERTEncoder, OneHotClassEmbedding, UnconditionalClassEmbedding
import torch
from models.pixelcnnpp import ConditionalPixelCNNpp
from utils.utils import sample_image, load_model
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument('--lr_decay', type=float, default=0.99,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_resnet", type=int, default=5, help="number of layers for the pixelcnn model")
parser.add_argument("--n_filters", type=int, default=160, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--use_cuda", type=int, default=1, help="use cuda if available")
parser.add_argument("--output_dir", type=str, default="outputs/pixelcnn", help="directory to store the sampled outputs")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--train_on_val", type=int, default=0, help="train on val set, useful for debugging")
parser.add_argument("--train", type=int, default=1, help="0 = eval, 1 = train, 2 = sample")
parser.add_argument("--model_checkpoint", type=str, default=None,
                    help="load model from checkpoint, model_checkpoint = path_to_your_pixel_cnn_model.pt")
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--dataset", type=str, default="cifar10", choices=["imagenet32", "cifar10"])
parser.add_argument("--conditioning", type=str, default="unconditional", choices=["unconditional", "one-hot", "bert"])

parser.add_argument("--n_row", type=int, default=4, help="size of sample image grid")
parser.add_argument("--sample_id", type=int, default=0, help="name of sample image")

def train(model, embedder, optimizer, scheduler,
          train_loader, val_loader, opt):
    print("TRAINING STARTS")
    for epoch in range(opt.n_epochs):
        model = model.train()
        loss_to_log = 0.0
        for i, (imgs, labels, captions) in enumerate(train_loader):
            start_batch = time.time()
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                condition_embd = embedder(labels, captions)

            optimizer.zero_grad()
            outputs = model.forward(imgs, condition_embd)
            loss = outputs['loss'].mean()
            loss.backward()
            optimizer.step()
            batches_done = epoch * len(train_loader) + i
            writer.add_scalar('train/bpd', loss / np.log(2), batches_done)
            loss_to_log += loss.item()
            if (i + 1) % opt.print_every == 0:
                loss_to_log = loss_to_log / (np.log(2) * opt.print_every)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [bpd: %f] [Time/batch %.3f]"
                    % (epoch + 1, opt.n_epochs, i + 1, len(train_loader), loss_to_log, time.time() - start_batch),
                    flush = True
                )
                loss_to_log = 0.0

            if (batches_done + 1) % opt.sample_interval == 0:
                print("sampling_images")
                model = model.eval()
                sample_image(model, embedder, opt.output_dir, n_row=opt.n_row,
                             batches_done=batches_done,
                             dataloader=val_loader, device=device)
        val_bpd = eval(model, embedder, val_loader)
        writer.add_scalar("val/bpd", val_bpd, (epoch + 1) * len(train_loader))

        torch.save(model.state_dict(),
                   os.path.join(opt.output_dir, 'models', 'epoch_{}.pt'.format(epoch)))

    scheduler.step()

def sample(model, embedder, batches_done, val_loader, device):
    print("sampling_images")
    model = model.eval()
    sample_image(model, embedder, opt.output_dir, n_row=opt.n_row,
                 batches_done=batches_done,
                 dataloader=val_loader, device=device)
    

def eval(model, embedder, test_loader):
    print("EVALUATING ON VAL")
    model = model.eval()
    bpd = 0.0
    for i, (imgs, labels, captions) in tqdm(enumerate(test_loader), total=len(test_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            condition_embd = embedder(labels, captions)
            outputs = model.forward(imgs, condition_embd)
            loss = outputs['loss'].mean()
            bpd += loss / np.log(2)
    bpd /= len(test_loader)
    print("VAL bpd : {}".format(bpd))
    return bpd


if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)

    print("loading dataset")
    if opt.dataset == "imagenet32":
        train_dataset = Imagenet32Dataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = Imagenet32Dataset(train=0, max_size=1 if opt.debug else -1)
    else:
        assert opt.dataset == "cifar10"
        train_dataset = CIFAR10Dataset(train=not opt.train_on_val, max_size=1 if opt.debug else -1)
        val_dataset = CIFAR10Dataset(train=0, max_size=1 if opt.debug else -1)

    print("creating dataloaders")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )

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

    generative_model = ConditionalPixelCNNpp(embd_size=encoder.embed_size, img_shape=train_dataset.image_shape,
                                             nr_resnet=opt.n_resnet, nr_filters=opt.n_filters,
                                             nr_logistic_mix=3 if train_dataset.image_shape[0] == 1 else 10)

    generative_model = generative_model.to(device)
    encoder = encoder.to(device)
    print("Models loaded on device")

    # Configure data loader

    print("dataloaders loaded")
    # Optimizers
    optimizer = torch.optim.Adam(generative_model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
    # create output directory

    os.makedirs(os.path.join(opt.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "tensorboard"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, "tensorboard"))

    # ----------
    #  Training
    # ----------
    if opt.train == 1:
        train(model=generative_model, embedder=encoder, optimizer=optimizer, scheduler=scheduler,
              train_loader=train_dataloader, val_loader=val_dataloader, opt=opt)
    else:
        assert opt.model_checkpoint is not None, 'no model checkpoint specified'
        print("Loading model from state dict...")
        load_model(opt.model_checkpoint, generative_model, device)
        print("Model loaded.")
        

    if opt.train == 0:
        eval(model=generative_model, embedder=encoder, test_loader=val_dataloader)
    if opt.train == 2:
        sample(model=generative_model, embedder=encoder, batches_done=opt.sample_id, val_loader=val_dataloader, device=device)
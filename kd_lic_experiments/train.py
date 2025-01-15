import os
import sys
import shutil
import random

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim

from compressai.datasets import Vimeo90kDataset
# from compressai.zoo.image import _load_model, bmshj2018_hyperprior, mbt2018_mean, mbt2018
from compressai.models.google import ScaleHyperprior

from tqdm.auto import tqdm
import wandb

from losses import AverageMeter

# Set seeds
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_bpp(out_net):
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net["likelihoods"].values()).item()


def save_checkpoint(
        checkpoint, is_best, path="checkpoint.pth.tar",
        best_path="checkpoint_best.pth.tar"):
    torch.save(checkpoint, path)
    if is_best:
        shutil.copyfile(path, best_path)


def make(config):
    # Create transformation
    load_transform = transforms.Compose(
        [transforms.RandomCrop((256,256)), transforms.ToTensor()]
    )

    # Create data sets
    if config.dataset == "Vimeo90K":
        train_dataset = Vimeo90kDataset(
            "/home/ids/fallemand-24/PRIM/data/vimeo/vimeo_triplet",
            split="train", transform=load_transform)
        validation_dataset = Vimeo90kDataset(
            "/home/ids/fallemand-24/PRIM/data/vimeo/vimeo_triplet",
            split="valid", transform=load_transform)
        test_dataset = validation_dataset
    else:
        raise ValueError("Invalid dataset")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=2,
        shuffle=True, pin_memory=(device == "cuda")
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, num_workers=2,
        shuffle=True, pin_memory=(device == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=2,
        shuffle=True, pin_memory=(device == "cuda")
    )

    # Create model
    model = ScaleHyperprior(config.N, config.M).to(device)

    # Create loss
    criterion = nn.MSELoss()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    
    return model, train_loader, validation_loader, test_loader, criterion, optimizer, lr_scheduler


def train_epoch(
        epoch, model, criterion, train_dataloader, optimizer, clip_max_norm=1.0):
    # Set-up
    model.train()
    device = next(model.parameters()).device

    n_examples = 0 # Number of examples processed
    for i, x in enumerate(train_dataloader):
        # Load batch
        x = x.to(device)

        # Forward pass
        output = model(x)
        loss = criterion(output["x_hat"], x)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimisation step
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # Update variables
        n_examples += len(x)

        # Logging
        if ((i + 1) % 10) == 0:
            wandb.log({"epoch": epoch, "loss": loss.item()}, step=n_examples)
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(x)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"loss: {loss.item():.3f}"
            )


def validation(data_loader, model, criterion):
    # Set-up
    model.eval()
    device = next(model.parameters()).device

    # Init measures
    avg_loss = AverageMeter()
    avg_psnr = AverageMeter()
    avg_msssim = AverageMeter()
    avg_bpp = AverageMeter()

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            # Load batch
            x = x.to(device)

            # Forward pass
            output = model(x)
            loss = criterion(output["x_hat"], x)

            # Update measures
            avg_loss.update(loss)
            avg_psnr.update(compute_psnr(x, output["x_hat"]))
            avg_msssim.update(compute_msssim(x, output["x_hat"]))
            avg_bpp.update(compute_bpp(output))

    # Logging
    log_dict = {
        "validation_loss": avg_loss.avg,
        "validation_psnr": avg_psnr.avg,
        "validation_msssim": avg_msssim.avg,
        "validation_bpp": avg_bpp.avg
    }
    wandb.log(log_dict)
    print(
        f"Validation: {f"{[f'{k} = {v:.6f}' for k, v in log_dict.items()]}"}"
    )

    return avg_loss.avg


def train(
        model, train_data_loader, validation_data_loader, criterion, optimizer,
        lr_scheduler, config):
    # Configure wandb
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    best_loss = float("inf") # Best loss
    for epoch in tqdm(range(config.epochs)):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Train one epoch
        train_epoch(epoch, model, criterion, train_data_loader, optimizer)

        # Validation
        loss = validation(validation_data_loader, model, criterion) # other data loader

        # Learning rate scheduler step
        lr_scheduler.step(loss)

        # Save model
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        save_checkpoint(
            checkpoint,
            is_best,
            os.path.join(config.save_path, "checkpoint.pth.tar"),
            os.path.join(config.save_path, "checkpoint_best.pth.tar")
        )


def test(model, data_loader, config):
    # Set-up
    student_device = next(model.parameters()).device
    checkpoint = torch.load(os.path.join(config.save_path, "checkpoint_best.pth.tar"),
                            weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(device)

    # Init measures
    criterion = nn.MSELoss()
    avg_loss = AverageMeter()
    avg_psnr = AverageMeter()
    avg_msssim = AverageMeter()
    avg_bpp = AverageMeter()

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            # Load batch
            x = x.to(student_device)

            # Forward pass
            output = model(x)
            loss = criterion(output["x_hat"], x)

            # Update measures
            avg_loss.update(loss)
            avg_psnr.update(compute_psnr(x, output["x_hat"]))
            avg_msssim.update(compute_msssim(x, output["x_hat"]))
            avg_bpp.update(compute_bpp(output))

    # Logging
    log_dict = {
        "test_loss": avg_loss.avg,
        "test_psnr": avg_psnr.avg,
        "test_msssim": avg_msssim.avg,
        "test_bpp": avg_bpp.avg
    }
    wandb.log(log_dict)
    print(
        f"Test: {f"{[f'{k} = {v:.6f}' for k, v in log_dict.items()]}"}"
    )


def model_pipeline(config):
    # Link to wandb project
    with wandb.init(project="bmshj2018_hyperprior_kd_experiments", config=config):
        # Access config
        config = wandb.config
        print(config)

        # Make model, data and optimizater
        model, train_loader, validation_loader, test_loader, criterion, optimizer, lr_scheduler = make(config)
        print(model)

        # Train model
        train(model, train_loader, validation_loader, criterion, optimizer, lr_scheduler, config)

        # Test model
        test(model, test_loader, config)

    return model


if __name__ == "__main__":
    job_id = sys.argv[1]

    # Experiment configuration
    config = dict(
        job_id=job_id,
        dataset="Vimeo90K",
        architecture="ScaleHyperprior",
        N=64,
        # N=128,
        M=192,
        epochs=1000,
        batch_size=128,
        learning_rate=1e-4,
        save_path=f"train_res/{job_id}"
    )

    model_pipeline(config)
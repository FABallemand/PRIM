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
from compressai.zoo.image import (
    model_urls,
    load_state_dict_from_url,
    load_pretrained,
    _load_model,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018
)

import wandb

from models import ScaleHyperprior, MeanScaleHyperprior, JointAutoregressiveHierarchicalPriors
from losses import AverageMeter, KDLoss

# Set seeds
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


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
    # Configure GPUs
    print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    teacher_device = torch.device("cuda:0")
    student_device = torch.device("cuda:1")

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
        shuffle=True, pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, num_workers=2,
        shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=2,
        shuffle=True, pin_memory=True
    )

    # Create models
    # teacher_model = bmshj2018_hyperprior(quality=5, # Best quality with (N, M) = (128, 192)
    #                                      pretrained=True).eval().to(teacher_device)
    url = model_urls["bmshj2018-hyperprior"]["mse"][5]
    state_dict = load_state_dict_from_url(url, progress=False)
    state_dict = load_pretrained(state_dict)
    teacher_model = ScaleHyperprior.from_state_dict(state_dict).eval().to(teacher_device)
    student_model = ScaleHyperprior(config.N_student, config.M).to(student_device)

    # Create loss
    criterion = KDLoss(latent=config.latent_loss)

    # Create optimizer
    optimizer = torch.optim.Adam(student_model.parameters(),
                                 lr=config.learning_rate)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    
    return teacher_model, student_model, train_loader, validation_loader, test_loader, criterion, optimizer, lr_scheduler


def train_epoch(
        epoch, teacher_model, student_model, criterion, train_loader,
        optimizer, clip_max_norm=1.0):
    # Set-up
    student_model.train()
    student_device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device

    for i, x in enumerate(train_loader):
        # Load batch
        teacher_x = x.to(teacher_device)
        student_x = x.to(student_device)

        # Forward pass
        teacher_output = teacher_model(teacher_x)
        student_output = student_model(student_x)
        loss, loss_dict = criterion(student_output["y_hat"], teacher_output["y_hat"].to(student_device),
                                    student_output["x_hat"], teacher_output["x_hat"].to(student_device),
                                    student_x)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimisation step
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

    # Logging
    log_dict = {
        "epoch": epoch
    }
    log_dict |= loss_dict
    wandb.log(log_dict, step=epoch)
    print(
        f"Epoch {epoch}: ["
        f"{i*len(x)}/{len(train_loader.dataset)}"
        f" ({100. * i / len(train_loader):.0f}%)]"
        f"{f"{[f'{k} = {v:.6f}' for k, v in loss_dict.items()]}"}"
    )


def validation(epoch, data_loader, teacher_model, student_model, criterion):
    # Set-up
    student_model.eval()
    student_device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device

    # Init measures
    avg_loss = AverageMeter()
    avg_latent_kd_loss = AverageMeter()
    avg_output_kd_loss = AverageMeter()
    avg_output_loss = AverageMeter()
    avg_psnr = AverageMeter()
    avg_msssim = AverageMeter()
    avg_bpp = AverageMeter()

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            # Load batch
            teacher_x = x.to(teacher_device)
            student_x = x.to(student_device)

            # Forward pass
            teacher_output = teacher_model(teacher_x)
            student_output = student_model(student_x)
            loss, loss_dict = criterion(student_output["y_hat"], teacher_output["y_hat"].to(student_device),
                                        student_output["x_hat"], teacher_output["x_hat"].to(student_device),
                                        student_x)
            # Update measures
            avg_loss.update(loss_dict["loss"])
            avg_latent_kd_loss.update(loss_dict["latent_kd_loss"])
            avg_output_kd_loss.update(loss_dict["output_kd_loss"])
            avg_output_loss.update(loss_dict["output_loss"])
            avg_psnr.update(compute_psnr(student_x, student_output["x_hat"]))
            avg_msssim.update(compute_msssim(student_x, student_output["x_hat"]))
            avg_bpp.update(compute_bpp(student_output))

    # Logging
    log_dict = {
        "epoch": epoch,
        "validation_loss": avg_loss.avg,
        "validation_latent_kd_loss": avg_latent_kd_loss.avg,
        "validation_output_kd_loss": avg_output_kd_loss.avg,
        "validation_output_loss": avg_output_loss.avg,
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
        teacher_model, student_model, train_data_loader, validation_data_loader,
        criterion, optimizer, lr_scheduler, config):
    # Configure wandb
    wandb.watch(student_model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    best_loss = float("inf") # Best loss
    for epoch in range(config.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Train one epoch
        train_epoch(epoch, teacher_model, student_model, criterion, train_data_loader, optimizer)

        # Validation
        loss = validation(epoch, validation_data_loader, teacher_model, student_model, criterion)

        # Learning rate scheduler step
        lr_scheduler.step(loss)

        # Save model
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        checkpoint = {
            "epoch": epoch,
            "state_dict": student_model.state_dict(),
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


def test(student_model, data_loader, config):
    # Set-up
    student_device = next(student_model.parameters()).device
    checkpoint = torch.load(os.path.join(config.save_path, "checkpoint_best.pth.tar"),
                            weights_only=True, map_location=torch.device("cpu"))
    student_model.load_state_dict(checkpoint["state_dict"])
    student_model = student_model.eval().to(student_device)

    # Init measures
    criterion = nn.MSELoss()
    avg_loss = AverageMeter()
    avg_psnr = AverageMeter()
    avg_msssim = AverageMeter()
    avg_bpp = AverageMeter()

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            # Load batch
            student_x = x.to(student_device)

            # Forward pass
            student_output = student_model(student_x)
            loss = criterion(student_output["x_hat"], student_x)

            # Update measures
            avg_loss.update(loss)
            avg_psnr.update(compute_psnr(student_output["x_hat"], student_x))
            avg_msssim.update(compute_msssim(student_output["x_hat"], student_x))
            avg_bpp.update(compute_bpp(student_output))

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
        teacher_model, student_model, train_loader, validation_loader, test_loader, criterion, optimizer, lr_scheduler = make(config)
        print(teacher_model)
        print(student_model)
        print(criterion)

        # Train model
        train(teacher_model, student_model, train_loader, validation_loader, criterion, optimizer, lr_scheduler, config)

        # Test model
        test(student_model, test_loader, config)

    return teacher_model, student_model


if __name__ == "__main__":
    job_id = sys.argv[1]

    # Experiment configuration
    config = dict(
        job_id=job_id,
        dataset="Vimeo90K",
        N_student=16,
        M=192,
        epochs=1000,
        batch_size=16,
        learning_rate=1e-4,
        latent_loss=True,
        save_path=f"train_res/{job_id}"
    )

    model_pipeline(config)
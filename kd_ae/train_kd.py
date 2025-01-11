import os
import sys
import shutil
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from compressai.datasets import Vimeo90kDataset
# from compressai.zoo.image import _load_model
# from compressai.models.google import ScaleHyperprior

from tqdm.auto import tqdm
import wandb

from models import ScaleHyperprior
from losses import AverageMeter, KDLoss

# Set seeds
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
        checkpoint, is_best, path="checkpoint.pth.tar",
        best_path="checkpoint_best.pth.tar"):
    print(path)
    torch.save(checkpoint, path)
    if is_best:
        print(best_path)
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

    # Create models
    teacher_model = ScaleHyperprior(config.teacher_N, config.M).to(device)
    student_model = ScaleHyperprior(config.student_N, config.M).to(device)

    # Create loss
    criterion = KDLoss(latent=config.latent_loss)

    # Create optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    
    return teacher_model, student_model, train_loader, validation_loader, test_loader, criterion, optimizer, lr_scheduler


def train_epoch(
        epoch, teacher_model, student_model, criterion, train_dataloader,
        optimizer, clip_max_norm=1.0):
    # Set-up
    student_model.train()
    device = next(student_model.parameters()).device

    n_examples = 0 # Number of examples processed
    for i, x in enumerate(train_dataloader):
        # Load batch
        x = x.to(device)

        # Forward pass
        teacher_output = teacher_model(x)
        student_output = student_model(x)
        loss, loss_dict = criterion(student_output["y_hat"], teacher_output["y_hat"],
                                    student_output["x_hat"], teacher_output["x_hat"],
                                    x)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimisation step
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Update variables
        n_examples += len(x)

        # Logging
        if ((i + 1) % 10) == 0:
            log_dict = {
                "epoch": epoch,
                "loss": loss.item()
            }
            log_dict |= loss_dict
            wandb.log(log_dict, step=n_examples)
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(x)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tLoss: {loss.item():.3f}"
                f"\t{f"{[f'{k} = {v:.3f}' for k, v in loss_dict.items()]}"}"
            )


def train(
        teacher_model, student_model, train_data_loader, validation_data_loader,
        criterion, optimizer, lr_scheduler, config):
    # Configure wandb
    wandb.watch(student_model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    best_loss = float("inf") # Best loss
    for epoch in tqdm(range(config.epochs)):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Train one epoch
        train_epoch(epoch, teacher_model, student_model, criterion, train_data_loader, optimizer)

        # Validation
        loss = validation(validation_data_loader, student_model, criterion) # other data loader

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
            os.path.join(config.save_path, "/checkpoint.pth.tar"),
            os.path.join(config.save_path, "/checkpoint_best.pth.tar")
        )


def validation(data_loader, model, criterion):
    # Set-up
    model.eval()
    device = next(model.parameters()).device

    avg_loss = AverageMeter()

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            # Load batch
            x = x.to(device)

            # Forward pass
            output = model(x)
            loss = criterion(output["x_hat"], x)

            # Update loss
            avg_loss.update(loss)

    print(f"Validation loss: {avg_loss.avg:.3f}")

    return avg_loss.avg


def test(model, data_loader, criterion, config):
    # Set-up
    device = next(model.parameters()).device
    checkpoint = torch.load(f"{config.save_path}/checkpoint_best.pth.tar",
                            weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(device)

    avg_loss = AverageMeter()

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            # Load batch
            x = x.to(device)

            # Forward pass
            output = model(x)
            loss = criterion(output["x_hat"], x)

            # Update loss
            avg_loss.update(loss)

    # Logging
    wandb.log({"test_loss": avg_loss.avg})
    print(f"Validation loss: {avg_loss.avg:.3f}")

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, x, f"{config.save_path}/model.onnx")
    wandb.save(f"{config.save_path}/model.onnx")


def model_pipeline(config):
    # Link to wandb project
    with wandb.init(project="bmshj2018_hyperprior_kd", config=config):
        # Access config
        config = wandb.config
        print(config)

        # Make model, data and optimizater
        teacher_model, student_model, train_loader, validation_loader, test_loader, criterion, optimizer, lr_scheduler = make(config)
        print(teacher_model)
        print(student_model)

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
        architecture="ScaleHyperprior",
        N_teacher=128,
        N_student=64,
        M=192,
        teacher_checkpoint="train_res/id/best_checkpoint.pth.tar",
        epochs=1000,
        batch_size=128,
        learning_rate=1e-4,
        latent_loss=False,
        save_path=f"train_res/{job_id}"
    )

    model_pipeline(config)
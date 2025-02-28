# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import Vimeo90kDataset
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

from models import TeacherAE, StudentAE


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


class KDLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.lmbda_1 = 0.2
        self.lmbda_2 = 0.2
        self.lmbda_3 = 0.6

        # self.latent_kd_loss = nn.KLDivLoss()
        self.latent_kd_loss = nn.MSELoss()
        self.output_kd_loss = nn.MSELoss()
        self.output_loss = nn.MSELoss()

    def forward(self, latent_input, latent_kd_target, input, kd_trarget, target):
        latent_kd_loss = self.lmbda_1 * self.latent_kd_loss(latent_input, latent_kd_target)
        kd_loss = self.lmbda_2 * self.output_kd_loss(input, kd_trarget)
        loss = self.lmbda_3 * self.output_loss(input, target)
        return latent_kd_loss + kd_loss + loss


def train_one_epoch(
    teacher_net, student_net, criterion, train_dataloader, optimizer, epoch, clip_max_norm
):
    teacher_device = next(teacher_net.parameters()).device

    student_net.train()
    student_device = next(student_net.parameters()).device

    for i, d in enumerate(train_dataloader):
        d_student = d.to(student_device)

        d_noisy = d + torch.randn_like(d) * 0.2
        d_teacher_noisy = d_noisy.to(teacher_device)
        d_student_noisy = d_noisy.to(student_device)

        optimizer.zero_grad()

        latent_teacher_out, teacher_out = teacher_net(d_teacher_noisy)
        latent_student_out, student_out = student_net(d_student_noisy)

        out_criterion = criterion(latent_student_out, latent_teacher_out,
                                  student_out, teacher_out, d_student)
        out_criterion.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), clip_max_norm)
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion.item():.3f}\n'
            )


def test_epoch(epoch, test_dataloader, teacher_net, student_net, criterion):
    teacher_device = next(teacher_net.parameters()).device

    student_net.eval()
    student_device = next(student_net.parameters()).device

    loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d_student = d.to(student_device)

            d_noisy = d + torch.randn_like(d) * 0.2
            d_teacher_noisy = d_noisy.to(teacher_device)
            d_student_noisy = d_noisy.to(student_device)

            latent_teacher_out, teacher_out = teacher_net(d_teacher_noisy)
            latent_student_out, student_out = student_net(d_student_noisy)

            out_criterion = criterion(latent_student_out, latent_teacher_out,
                                      student_out, teacher_out, d_student)

            loss.update(out_criterion)

    print(
        f"Test epoch {epoch}: "
        f"Loss: {loss.avg:.3f}"
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_filename="checkpoint_best.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--savepath", type=str, help="Path to save checkpoint")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--teacher-checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = Vimeo90kDataset(args.dataset, split="train", transform=train_transforms)
    test_dataset = Vimeo90kDataset(args.dataset, split="valid", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    teacher_net = TeacherAE()
    checkpoint = torch.load(args.teacher_checkpoint,
        weights_only=True, map_location=torch.device("cpu"))
    teacher_net.load_state_dict(checkpoint["state_dict"])
    teacher_net = teacher_net.eval().to(device)

    student_net = StudentAE()
    student_net = student_net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        teacher_net = CustomDataParallel(teacher_net)
        student_net = CustomDataParallel(student_net)

    optimizer = torch.optim.Adam(student_net.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = KDLoss()

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        student_net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            teacher_net,
            student_net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, teacher_net, student_net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            if args.savepath:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": student_net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    args.savepath + "/checkpoint.pth.tar",
                    args.savepath + "/checkpoint_best.pth.tar"
                )
            else:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": student_net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                )


if __name__ == "__main__":
    main(sys.argv[1:])
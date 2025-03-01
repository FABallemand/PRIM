import os
import time
from datetime import datetime
import json

import math
import numpy as np
import scipy.interpolate

import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_msssim import ms_ssim

from compressai.zoo.image import (
    model_urls,
    load_state_dict_from_url,
    load_pretrained,
    bmshj2018_hyperprior
)

from fvcore.nn import FlopCountAnalysis
from zeus.monitor import ZeusMonitor
# from zeus.device.cpu import get_current_cpu_index
import pynvml

import matplotlib.pyplot as plt
from PIL import Image

from models import ScaleHyperprior

# Set-up PyTorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set-up matplotlib
plt.rcParams["axes.prop_cycle"] = plt.rcParams["axes.prop_cycle"][1:]

# Create output directory
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"/home/ids/fallemand-24/PRIM/kd_lic_experiments/test_res/{time_stamp}"
os.makedirs(output_folder)

###############################################################################
## Utils ######################################################################
###############################################################################

def model_nb_param(model):
    return sum(p.numel() for p in model.parameters())


def model_memory_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


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


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff

###############################################################################
## Networks ###################################################################
###############################################################################

# Networks parameters
N = 64
M = 192

# Load networks

# RD loss
# lmbdas = [0.0250, 0.0018, 0.0035, 0.0067, 0.0130, 0.0250]
# ids = [None, 280392, 281662, 281976, 281979, 274461]
# teacher_quality = 5

# networks = {
#     "teacher": None,
#     "student_1": None,
#     "student_2": None,
#     "student_3": None,
#     "student_4": None,
#     "student_5": None,
# }

# RD loss
lmbdas = [0.0018, 0.0018, 0.0067, 0.025]
ids = [None, 289751, 289745, 289742]
teacher_quality = 1

networks = {
    "teacher": None,
    "student_1": None,
    "student_2": None,
    "student_3": None,
}

for name, id_ in zip(networks.keys(), ids):
    if name == "teacher":
        url = model_urls["bmshj2018-hyperprior"]["mse"][teacher_quality]
        state_dict = load_state_dict_from_url(url, progress=False)
        state_dict = load_pretrained(state_dict)
        net = ScaleHyperprior.from_state_dict(state_dict).eval().to(DEVICE)
    else:
        net = ScaleHyperprior(N, M)
        checkpoint = torch.load(f"train_res/{id_}/checkpoint_best.pth.tar",
            weights_only=True, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["state_dict"])
    networks[name] = net.eval().to(DEVICE)

# Create dict for average metrics
avg_metrics = {}
for name, net in networks.items():
    avg_metrics[name] = {
            "params": model_nb_param(net),
            "memory": model_memory_size(net),
            "flops": None,
            "inference-time": [],
            "zeus-energy-time": [],
            "zeus-energy": [],
            "pynvml-energy-time": [],
            "pynvml-energy": [],
            "mse": [],
            "psnr": [],
            "ms-ssim": [],
            "bit-rate": [],
        }

# Load pre-trained networks
pretrained_networks = {}

for quality in range(1, 9):
    net = bmshj2018_hyperprior(quality=quality,
                               pretrained=True).eval().to(DEVICE)

    pretrained_networks[f"{quality}"] = net

# Create dict for pre-trained average metrics
pretrained_avg_metrics = {}
for name, net in pretrained_networks.items():
    pretrained_avg_metrics[name] = {
            "params": model_nb_param(net),
            "memory": model_memory_size(net),
            "flops": None,
            "inference-time": [],
            "zeus-energy-time": [],
            "zeus-energy": [],
            "pynvml-energy-time": [],
            "pynvml-energy": [],
            "mse": [],
            "psnr": [],
            "ms-ssim": [],
            "bit-rate": [],
        }

###############################################################################
## Data #######################################################################
###############################################################################

# Dataset path
DATASET_PATH = "/home/ids/fallemand-24/PRIM/data/kodak"
# DATASET_PATH = "/home/ids/fallemand-24/PRIM/data/clic/clic_validation"
# DATASET_PATH = "/home/ids/fallemand-24/PRIM/data/clic/clic_test"

# Dataset name
dataset_name = DATASET_PATH.split("/")[-1]

# Images path
dataset_imgs = [p for p in os.listdir(DATASET_PATH) if p.endswith(".png")]

###############################################################################
## Inference and Metrics (single image) #######################################
###############################################################################

# Compute flops
COMPUTE_FLOP = True

# Criterion
criterion = nn.MSELoss()

# Iterate over images
for img_name in dataset_imgs:
    # Load image
    img = Image.open(os.path.join(DATASET_PATH, img_name)).convert("RGB")
    if dataset_name == "clic":
        img = img.crop((0, 0, 768, 512)) # For CLIC dataset
    x = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    img_name = img_name.split(".")[0]

    # FLOPs computation
    if COMPUTE_FLOP:
        # Networks flops computation
        with torch.no_grad():
            for name, net in networks.items():
                flops = FlopCountAnalysis(net, x)
                avg_metrics[name]["flops"] = flops.total()

        # Pre-trained networks flops computation
        with torch.no_grad():
            for name, net in pretrained_networks.items():
                flops = FlopCountAnalysis(net, x)
                pretrained_avg_metrics[name]["flops"] = flops.total()

        # Turn off FLOPs computation
        COMPUTE_FLOP = False

    # Inference and metrics for networks
    outputs = {}
    metrics = {}
    with torch.no_grad():
        for name, net in networks.items():
            # Run inference
            start = time.time()
            out = net(x)
            stop = time.time()
            out["x_hat"].clamp_(0, 1)

            # Save output
            outputs[name] = out

            # Compute metrics
            metrics[name] = {
                "inference-time": stop - start,
                "mse": criterion(out["x_hat"], x).item(),
                "psnr": compute_psnr(out["x_hat"], x),
                "ms-ssim": compute_msssim(out["x_hat"], x),
                "bit-rate": compute_bpp(out),
            }
            avg_metrics[name]["inference-time"].append(metrics[name]["inference-time"])
            avg_metrics[name]["mse"].append(metrics[name]["mse"])
            avg_metrics[name]["psnr"].append(metrics[name]["psnr"])
            avg_metrics[name]["ms-ssim"].append(metrics[name]["ms-ssim"])
            avg_metrics[name]["bit-rate"].append(metrics[name]["bit-rate"])

    # Inference and metrics for pre-trained networks
    pretrained_outputs = {}
    pretrained_metrics = {}
    with torch.no_grad():
        for name, net in pretrained_networks.items():
            # Run inference
            start = time.time()
            out = net(x)
            stop = time.time()
            out["x_hat"].clamp_(0, 1)

            pretrained_outputs[name] = out

            pretrained_metrics[name] = {
                "inference-time": stop - start,
                "mse": criterion(out["x_hat"], x).item(),
                "psnr": compute_psnr(out["x_hat"], x),
                "ms-ssim": compute_msssim(out["x_hat"], x),
                "bit-rate": compute_bpp(out),
            }

            pretrained_avg_metrics[name]["inference-time"].append(pretrained_metrics[name]["inference-time"])
            pretrained_avg_metrics[name]["mse"].append(pretrained_metrics[name]["mse"])
            pretrained_avg_metrics[name]["psnr"].append(pretrained_metrics[name]["psnr"])
            pretrained_avg_metrics[name]["ms-ssim"].append(pretrained_metrics[name]["ms-ssim"])
            pretrained_avg_metrics[name]["bit-rate"].append(pretrained_metrics[name]["bit-rate"])

    # Save metrics
    all_metrics = metrics | pretrained_metrics
    with open(os.path.join(output_folder,
                           f"metrics_{dataset_name}_{img_name}.json"),
                           "w", encoding="utf-8") as f:
        json.dump(all_metrics, f)

###############################################################################
## Plots (single image) #######################################################
###############################################################################

    reconstructions = {name: transforms.ToPILImage()(out["x_hat"].squeeze())
                       for name, out in outputs.items()}

    diffs = [torch.mean((out["x_hat"] - x).abs(), axis=1).squeeze()
             for out in outputs.values()]

    pretrained_reconstructions = {name: transforms.ToPILImage()(out["x_hat"].squeeze())
                                  for name, out in pretrained_outputs.items()}

    pretrained_diffs = [torch.mean((out["x_hat"] - x).abs(), axis=1).squeeze()
                        for out in pretrained_outputs.values()]

    # Compare networks
    n_rows = math.ceil((len(reconstructions) + 1) / 3)
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    for ax in axs.ravel():
        ax.axis("off")

    axs.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
    axs.ravel()[0].title.set_text("Original")

    for i, (name, rec) in enumerate(reconstructions.items()):
        axs.ravel()[i + 1].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
        axs.ravel()[i + 1].title.set_text(name)

    fig.tight_layout()

    plt.savefig(os.path.join(output_folder,
                             f"networks_{dataset_name}_{img_name}.png"))
    plt.close()

    # Compare pre-trained networks
    # n_rows = math.ceil((len(reconstructions) + 1) / 3)
    # n_cols = 3
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    # for ax in axs.ravel():
    #     ax.axis("off")

    # axs.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
    # axs.ravel()[0].title.set_text("Original")

    # for i, (name, rec) in enumerate(pretrained_reconstructions.items()):
    #     axs.ravel()[i + 1].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
    #     axs.ravel()[i + 1].title.set_text(name)

    # fig.tight_layout()

    # plt.savefig(os.path.join(output_folder,
    #                          f"pretrained_networks_{dataset_name}_{img_name}.png"))
    # plt.close()

    # Plot rate-distortion curves
    # fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # brs = [m["bit-rate"] for _, m in metrics.items()]
    # pretrained_brs = [m["bit-rate"] for _, m in pretrained_metrics.items()]

    # psnrs = [m["psnr"] for _, m in metrics.items()]
    # # axs[0].plot(brs, psnrs, "red", linestyle="--", linewidth=1, label="proposed")

    # pretrained_psnrs = [m["psnr"] for _, m in pretrained_metrics.items()]
    # axs[0].plot(pretrained_brs, pretrained_psnrs, "blue", linestyle="--",
    #             linewidth=1, label="pre-trained")

    # msssim = [-10*np.log10(1-m["ms-ssim"]) for _, m in metrics.items()]
    # # axs[1].plot(brs, msssim, "red", linestyle="--", linewidth=1, label="proposed")

    # pretrained_msssim = [-10*np.log10(1-m["ms-ssim"]) for _, m in pretrained_metrics.items()]
    # axs[1].plot(pretrained_brs, pretrained_msssim, "blue", linestyle="--",
    #             linewidth=1, label="pre-trained")

    # for name, m in pretrained_metrics.items():
    #     axs[0].plot(m["bit-rate"], m["psnr"], "o", color="blue")
    #     axs[0].grid(True)
    #     axs[0].set_ylabel("PSNR [dB]")
    #     axs[0].set_xlabel("Bit rate [bpp]")
    #     axs[0].title.set_text("PSNR comparison")

    #     axs[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]), "o", color="blue")
    #     axs[1].grid(True)
    #     axs[1].set_ylabel("MS-SSIM [dB]")
    #     axs[1].set_xlabel("Bit rate [bpp]")
    #     axs[1].title.set_text("MS-SSIM (log) comparison")

    # for name, m in metrics.items():
    #     axs[0].plot(m["bit-rate"], m["psnr"],
    #                  "s" if name == "teacher" else "o", label=name)
    #     axs[0].grid(True)
    #     axs[0].set_ylabel("PSNR [dB]")
    #     axs[0].set_xlabel("Bit rate [bpp]")
    #     axs[0].title.set_text("PSNR comparison")

    #     axs[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]),
    #                  "s" if name == "teacher" else "o", label=name)
    #     axs[1].grid(True)
    #     axs[1].set_ylabel("MS-SSIM [dB]")
    #     axs[1].set_xlabel("Bit rate [bpp]")
    #     axs[1].title.set_text("MS-SSIM (log) comparison")

    # axs[0].legend(loc="best")
    # axs[1].legend(loc="best")

    # fig.tight_layout()

    # plt.savefig(os.path.join(output_folder,
    #                          f"curve_{dataset_name}_{img_name}.png"))
    # plt.close()

###############################################################################
## Metrics (average) ##########################################################
###############################################################################

# Zeus energy consumption
# zeus_monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
zeus_monitor = ZeusMonitor(approx_instant_energy=True)

# Iterate over networks
for name, net in networks.items():
    # Iterate over images
    zeus_monitor.begin_window("inference")
    for img_name in dataset_imgs:
        # Load image
        img = Image.open(os.path.join(DATASET_PATH, img_name)).convert("RGB")
        if dataset_name == "clic":
            img = img.crop((0, 0, 768, 512)) # For CLIC dataset
        x = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

        # Inference
        out = net(x)
    mes = zeus_monitor.end_window("inference")

    avg_metrics[name]["zeus-energy-time"] = mes.time
    avg_metrics[name]["zeus-energy"] = mes.total_energy

# Iterate over pre-trained networks
for name, net in pretrained_networks.items():
    # Iterate over images
    zeus_monitor.begin_window("inference")
    for img_name in dataset_imgs:
        # Load image
        img = Image.open(os.path.join(DATASET_PATH, img_name)).convert("RGB")
        if dataset_name == "clic":
            img = img.crop((0, 0, 768, 512)) # For CLIC dataset
        x = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

        # Inference
        out = net(x)
    mes = zeus_monitor.end_window("inference")

    pretrained_avg_metrics[name]["zeus-energy-time"] = mes.time
    pretrained_avg_metrics[name]["zeus-energy"] = mes.total_energy

# pynvml energy consumption
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # First GPU

# Iterate over networks
for name, net in networks.items():
    start_time = time.time()
    start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    # Iterate over images
    for img_name in dataset_imgs:
        # Load image
        img = Image.open(os.path.join(DATASET_PATH, img_name)).convert("RGB")
        if dataset_name == "clic":
            img = img.crop((0, 0, 768, 512)) # For CLIC dataset
        x = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

        # Inference
        out = net(x)
    torch.cuda.synchronize()  # Synchronizes CPU and GPU time.
    elapsed_time = time.time() - start_time
    consumed_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy

    avg_metrics[name]["pynvml-energy-time"] = elapsed_time
    avg_metrics[name]["pynvml-energy"] = consumed_energy * 1000 # Convert to J

# Iterate over pre-trained networks
for name, net in pretrained_networks.items():
    start_time = time.time()
    start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    # Iterate over images
    for img_name in dataset_imgs:
        # Load image
        img = Image.open(os.path.join(DATASET_PATH, img_name)).convert("RGB")
        if dataset_name == "clic":
            img = img.crop((0, 0, 768, 512)) # For CLIC dataset
        x = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

        # Inference
        out = net(x)
    torch.cuda.synchronize()  # Synchronizes CPU and GPU time.
    elapsed_time = time.time() - start_time
    consumed_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy

    pretrained_avg_metrics[name]["pynvml-energy-time"] = elapsed_time
    pretrained_avg_metrics[name]["pynvml-energy"] = consumed_energy * 1000 # Convert to J

# Compute average metrics
for name in networks:
    avg_metrics[name]["inference-time"] = np.average(avg_metrics[name]["inference-time"])
    avg_metrics[name]["mse"] = np.average(avg_metrics[name]["mse"])
    avg_metrics[name]["psnr"] = np.average(avg_metrics[name]["psnr"])
    avg_metrics[name]["ms-ssim"] = np.average(avg_metrics[name]["ms-ssim"])
    avg_metrics[name]["bit-rate"] = np.average(avg_metrics[name]["bit-rate"])

# Compute pre-trained average metrics
for name in pretrained_networks:
    pretrained_avg_metrics[name]["inference-time"] = np.average(pretrained_avg_metrics[name]["inference-time"])
    pretrained_avg_metrics[name]["mse"] = np.average(pretrained_avg_metrics[name]["mse"])
    pretrained_avg_metrics[name]["psnr"] = np.average(pretrained_avg_metrics[name]["psnr"])
    pretrained_avg_metrics[name]["ms-ssim"] = np.average(pretrained_avg_metrics[name]["ms-ssim"])
    pretrained_avg_metrics[name]["bit-rate"] = np.average(pretrained_avg_metrics[name]["bit-rate"])

# Save average metrics
all_avg_metrics = {"proposed": avg_metrics, "pretrained": pretrained_avg_metrics}
with open(os.path.join(output_folder,
                       f"avg_metrics_{dataset_name}.json"),
                       "w", encoding="utf-8") as f:
    json.dump(all_avg_metrics, f)

# Retrieve average metrics as lists
brs = [m["bit-rate"] for _, m in avg_metrics.items()]
psnrs = [m["psnr"] for _, m in avg_metrics.items()]
msssim = [-10*np.log10(1-m["ms-ssim"]) for _, m in avg_metrics.items()]
pretrained_brs = [m["bit-rate"] for _, m in pretrained_avg_metrics.items()]
pretrained_psnrs = [m["psnr"] for _, m in pretrained_avg_metrics.items()]
pretrained_msssim = [-10*np.log10(1-m["ms-ssim"]) for _, m in pretrained_avg_metrics.items()]

# Compute average BD metrics
avg_bd_metrics = {}
avg_bd_metrics["bd_rate"] = BD_RATE(pretrained_brs, pretrained_psnrs, brs, psnrs)
avg_bd_metrics["bd_psnr"] = BD_PSNR(pretrained_brs, pretrained_psnrs, brs, psnrs)

# Save average BD metrics
with open(os.path.join(output_folder,
                       f"avg_bd_metrics_{dataset_name}.json"),
                       "w", encoding="utf-8") as f:
    json.dump(avg_bd_metrics, f)

###############################################################################
## Plots (average) ############################################################
###############################################################################

# Plot average rate-distortion curves
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

axs[0].plot(pretrained_brs[:-3], pretrained_psnrs[:-3], "blue", linestyle="--",
            linewidth=1, label="pre-trained")
axs[0].plot(brs[1:], psnrs[1:], "red", linestyle="--", linewidth=1,
            label=f"proposed\nBD-Rate: {avg_bd_metrics["bd_rate"]:.2f} %\nBD-PSNR: {avg_bd_metrics["bd_psnr"]:.2f} dB")
axs[1].plot(pretrained_brs[:-3], pretrained_msssim[:-3], "blue", linestyle="--",
            linewidth=1, label="pre-trained")
axs[1].plot(brs[1:], msssim[1:], "red", linestyle="--", linewidth=1, label="proposed")

for name, m in pretrained_avg_metrics.items():
    if name in ["1", "2", "3", "4", "5"]:
        axs[0].plot(m["bit-rate"], m["psnr"], "o", color="blue")
        axs[0].grid(True)
        axs[0].set_ylabel("PSNR [dB]")
        axs[0].set_xlabel("Bit rate [bpp]")
        axs[0].title.set_text("PSNR comparison")

        axs[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]), "o", color="blue")
        axs[1].grid(True)
        axs[1].set_ylabel("MS-SSIM [dB]")
        axs[1].set_xlabel("Bit rate [bpp]")
        axs[1].title.set_text("MS-SSIM (log) comparison")

for name, m in avg_metrics.items():
    axs[0].plot(m["bit-rate"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Bit rate [bpp]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]),
                 "s" if name == "teacher" else "o", label=name)
    axs[1].grid(True)
    axs[1].set_ylabel("MS-SSIM [dB]")
    axs[1].set_xlabel("Bit rate [bpp]")
    axs[1].title.set_text("MS-SSIM (log) comparison")

axs[0].legend(loc="best")
axs[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_rd_curve_{dataset_name}.png"))
plt.close()

# Plot number of parameters and RD performance
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

params = [m["params"] for _, m in avg_metrics.items()]
pretrained_params = [m["params"] for _, m in pretrained_avg_metrics.items()]

axs[0].plot(pretrained_params, pretrained_psnrs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")
axs[1].plot(pretrained_params, pretrained_brs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axs[0].plot(m["params"], m["psnr"], "o", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Number of parameters")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["params"], m["bit-rate"], "o", color="blue")
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Number of parameters")
    axs[1].title.set_text("Bit rate comparison")

for name, m in avg_metrics.items():
    axs[0].plot(m["params"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Number of parameters")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["params"], m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Number of parameters")
    axs[1].title.set_text("Bit rate comparison")

axs[0].legend(loc="best")
axs[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_param_curve_{dataset_name}.png"))
plt.close()

# Plot inference time and RD performance
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

times = [m["inference-time"] for _, m in avg_metrics.items()]
pretrained_times = [m["inference-time"] for _, m in pretrained_avg_metrics.items()]

axs[0].plot(pretrained_times, pretrained_psnrs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")
axs[1].plot(pretrained_times, pretrained_brs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axs[0].plot(m["inference-time"], m["psnr"], "o", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Inference time per frame [s]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["inference-time"], m["bit-rate"], "o", color="blue")
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Inference time per frame [s]")
    axs[1].title.set_text("Bit rate comparison")

for name, m in avg_metrics.items():
    axs[0].plot(m["inference-time"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Inference time [s]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["inference-time"], m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Inference time per frame [s]")
    axs[1].title.set_text("Bit rate comparison")

axs[0].legend(loc="best")
axs[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_time_curve_{dataset_name}.png"))
plt.close()

# Plot zeus-energy-time and RD performance
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

zeus_energy_times = [m["zeus-energy-time"] / len(dataset_imgs) for _, m in avg_metrics.items()]
zeus_pretrained_energy_times = [m["zeus-energy-time"] / len(dataset_imgs) for _, m in pretrained_avg_metrics.items()]

axs[0].plot(zeus_pretrained_energy_times, pretrained_psnrs, "blue",
            linestyle="--", linewidth=1, label="pre-trained")
axs[1].plot(zeus_pretrained_energy_times, pretrained_brs, "blue",
            linestyle="--", linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axs[0].plot(m["zeus-energy-time"] / len(dataset_imgs), m["psnr"], "o", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Time per frame [s]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["zeus-energy-time"] / len(dataset_imgs), m["bit-rate"], "o", color="blue")
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Time per frame [s]")
    axs[1].title.set_text("Bit rate comparison")

for name, m in avg_metrics.items():
    axs[0].plot(m["zeus-energy-time"] / len(dataset_imgs), m["psnr"],
                "s" if name == "teacher" else "o", label=name)
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Time per frame [s]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["zeus-energy-time"] / len(dataset_imgs), m["bit-rate"],
                "s" if name == "teacher" else "o", label=name)
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Time per frame [s]")
    axs[1].title.set_text("Bit rate comparison")

axs[0].legend(loc="best")
axs[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_zeus_energy_time_curve_{dataset_name}.png"))
plt.close()

# Plot zeus-energy and RD performance
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

zeus_energies = [m["zeus-energy"] / len(dataset_imgs) for _, m in avg_metrics.items()]
zeus_pretrained_energies = [m["zeus-energy"] / len(dataset_imgs) for _, m in pretrained_avg_metrics.items()]

axs[0].plot(zeus_pretrained_energies, pretrained_psnrs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")
axs[1].plot(zeus_pretrained_energies, pretrained_brs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axs[0].plot(m["zeus-energy"] / len(dataset_imgs), m["psnr"], "o", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Energy per frame [J]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["zeus-energy"] / len(dataset_imgs), m["bit-rate"], "o", color="blue")
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Energy per frame [J]")
    axs[1].title.set_text("Bit rate comparison")

for name, m in avg_metrics.items():
    axs[0].plot(m["zeus-energy"] / len(dataset_imgs), m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Energy per frame [J]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["zeus-energy"] / len(dataset_imgs), m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Energy per frame [J]")
    axs[1].title.set_text("Bit rate comparison")

axs[0].legend(loc="best")
axs[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_zeus_energy_curve_{dataset_name}.png"))
plt.close()

# Plot pynvml-energy-time and RD performance
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

pynvml_energy_times = [m["pynvml-energy-time"] / len(dataset_imgs) for _, m in avg_metrics.items()]
pynvml_pretrained_energy_times = [m["pynvml-energy-time"] / len(dataset_imgs) for _, m in pretrained_avg_metrics.items()]

axs[0].plot(pynvml_pretrained_energy_times, pretrained_psnrs, "blue",
            linestyle="--", linewidth=1, label="pre-trained")
axs[1].plot(pynvml_pretrained_energy_times, pretrained_brs, "blue",
            linestyle="--", linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axs[0].plot(m["pynvml-energy-time"] / len(dataset_imgs), m["psnr"], "o", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Time per frame [s]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["pynvml-energy-time"] / len(dataset_imgs), m["bit-rate"], "o", color="blue")
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Time per frame [s]")
    axs[1].title.set_text("Bit rate comparison")

for name, m in avg_metrics.items():
    axs[0].plot(m["pynvml-energy-time"] / len(dataset_imgs), m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Time per frame [s]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["pynvml-energy-time"] / len(dataset_imgs), m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Time per frame [s]")
    axs[1].title.set_text("Bit rate comparison")

axs[0].legend(loc="best")
axs[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_pynvml_energy_time_curve_{dataset_name}.png"))
plt.close()

# Plot pynvml-energy and RD performance
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

pynvml_energies = [m["pynvml-energy"] / len(dataset_imgs) for _, m in avg_metrics.items()]
pynvml_pretrained_energies = [m["pynvml-energy"] / len(dataset_imgs) for _, m in pretrained_avg_metrics.items()]

axs[0].plot(pynvml_pretrained_energies, pretrained_psnrs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")
axs[1].plot(pynvml_pretrained_energies, pretrained_brs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axs[0].plot(m["pynvml-energy"] / len(dataset_imgs), m["psnr"], "o", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Energy per frame [J]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["pynvml-energy"] / len(dataset_imgs), m["bit-rate"], "o", color="blue")
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Energy per frame [J]")
    axs[1].title.set_text("Bit rate comparison")

for name, m in avg_metrics.items():
    axs[0].plot(m["pynvml-energy"] / len(dataset_imgs), m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("Energy per frame [J]")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["pynvml-energy"] / len(dataset_imgs), m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("Energy per frame [J]")
    axs[1].title.set_text("Bit rate comparison")

axs[0].legend(loc="best")
axs[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_pynvml_energy_curve_{dataset_name}.png"))
plt.close()

# Plot FLOPs and RD performance
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

energies = [m["flops"] for _, m in avg_metrics.items()]
pretrained_energies = [m["flops"] for _, m in pretrained_avg_metrics.items()]

axs[0].plot(pretrained_energies, pretrained_psnrs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")
axs[1].plot(pretrained_energies, pretrained_brs, "blue", linestyle="--",
            linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axs[0].plot(m["flops"], m["psnr"], "o", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("FLOPs")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["flops"], m["bit-rate"], "o", color="blue")
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("FLOPs")
    axs[1].title.set_text("Bit rate comparison")

for name, m in avg_metrics.items():
    axs[0].plot(m["flops"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axs[0].grid(True)
    axs[0].set_ylabel("PSNR [dB]")
    axs[0].set_xlabel("FLOPs")
    axs[0].title.set_text("PSNR comparison")

    axs[1].plot(m["flops"], m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axs[1].grid(True)
    axs[1].set_ylabel("Bit rate [bpp]")
    axs[1].set_xlabel("FLOPs")
    axs[1].title.set_text("Bit rate comparison")

axs[0].legend(loc="best")
axs[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_flops_curve_{dataset_name}.png"))
plt.close()

# Plot mse and lambda
fig, axs = plt.subplots(1, 1, figsize=(6, 5))

axs.plot(lmbdas, [m["mse"] for _, m in avg_metrics.items()], "red", linestyle="--", linewidth=1)
for i, (name, m) in enumerate(avg_metrics.items()):
    axs.plot(lmbdas[i], m["mse"],
              "s" if name == "teacher" else "o", label=name)
axs.grid(True)
axs.set_ylabel("MSE")
axs.set_xlabel("Number of channels")
axs.title.set_text("MSE comparison")
axs.legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_mse_curve_{dataset_name}.png"))
plt.close()

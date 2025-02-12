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
    _load_model,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018
)

from fvcore.nn import FlopCountAnalysis
from zeus.monitor import ZeusMonitor
from zeus.device.cpu import get_current_cpu_index

import matplotlib.pyplot as plt
from PIL import Image

from models import ScaleHyperprior

# Set-up PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set-up matplotlib
plt.rcParams["axes.prop_cycle"] = plt.rcParams["axes.prop_cycle"][1:]

# Create output directory
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"/home/ids/fallemand-24/PRIM/kd_lic_experiments/test_res/{time_stamp}"
os.makedirs(output_folder)

###############################################################################
## Utils ######################################################################
###############################################################################

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
M = 192

# Load networks

# Wrong set of parameters
# Ns = [128, 16, 32, 64, 96, 112]
# ids = [None, 258263, 258258, 258259, 258262]

# No RD loss
# Ns = [128, 16, 32, 64, 96, 112]
# ids = [None, 259782, 259783, 259784, 259785, 259786]

# No RD loss
# Ns = [128, 16, 32, 64, 96, 112, 112]
# ids = [None, 259782, 259783, 259784, 259785, 259786, 261095]

# RD loss
Ns = [128, 16, 32, 64, 96, 112, 112]
ids = [None, 263674, 274457, 274461, 274464, 263691]

networks = {
    "teacher": None,
    "student_16": None,
    "student_32": None,
    "student_64": None,
    "student_96": None,
    "student_112": None,
    # "student_112_kl": None,
}

for name, N, id in zip(networks.keys(), Ns, ids):
    if name == "teacher":
        url = model_urls["bmshj2018-hyperprior"]["mse"][5]
        state_dict = load_state_dict_from_url(url, progress=False)
        state_dict = load_pretrained(state_dict)
        net = ScaleHyperprior.from_state_dict(state_dict).eval().to(device)
    else:
        net = ScaleHyperprior(N, M)
        checkpoint = torch.load(f"train_res/{id}/checkpoint_best.pth.tar",
            weights_only=True, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["state_dict"])
    networks[name] = net.eval().to(device)

# Create dict for average metrics
avg_metrics = {}
for name in networks.keys():
    avg_metrics[name] = {
            "params": sum(p.numel() for p in networks[name].parameters()),
            "flops": None,
            "inference-time": [],
            "energy-time": [],
            "energy": [],
            "mse": [],
            "psnr": [],
            "ms-ssim": [],
            "bit-rate": [],
        }
    
# Load pre-trained networks
pretrained_networks = {}

for quality in range(1, 9):
    net = bmshj2018_hyperprior(quality=quality,
                               pretrained=True).eval().to(device)

    pretrained_networks[f"{quality}"] = net

# Create dict for pre-trained average metrics
pretrained_avg_metrics = {}
for name in pretrained_networks.keys():
    pretrained_avg_metrics[name] = {
            "params": sum(p.numel() for p in pretrained_networks[name].parameters()),
            "flops": None,
            "inference-time": [],
            "energy-time": [],
            "energy": [],
            "mse": [],
            "psnr": [],
            "ms-ssim": [],
            "bit-rate": [],
        }
    
###############################################################################
## Data #######################################################################
###############################################################################

# Dataset path
dataset_path = "/home/ids/fallemand-24/PRIM/data/kodak"
# dataset_path = "/home/ids/fallemand-24/PRIM/data/clic/clic_validation"
# dataset_path = "/home/ids/fallemand-24/PRIM/data/clic/clic_test"

# Dataset name
dataset_name = dataset_path.split("/")[-1]

# Images path
dataset_imgs = [p for p in os.listdir(dataset_path) if p.endswith(".png")]

###############################################################################
## Inference and Metrics (single image) #######################################
###############################################################################

# Compute flops
compute_flops = True

# Energy consumption monitor
# monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
monitor = ZeusMonitor(approx_instant_energy=True)

# Criterion
criterion = nn.MSELoss()

# Iterate over images
for img_name in dataset_imgs:
    # Load image
    img = Image.open(os.path.join(dataset_path, img_name)).convert("RGB")
    if dataset_name == "clic":
        img = img.crop((0, 0, 768, 512)) # For CLIC dataset
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    img_name = img_name.split(".")[0]

    # FLOPs computation
    if compute_flops:
        # Networks flops computation
        with torch.no_grad():
            for model in networks.keys():
                flops = FlopCountAnalysis(networks[model], x)
                avg_metrics[model]["flops"] = flops.total()

        # Pre-trained networks flops computation
        with torch.no_grad():
            for model in pretrained_networks.keys():
                flops = FlopCountAnalysis(pretrained_networks[model], x)
                pretrained_avg_metrics[model]["flops"] = flops.total()

        # Turn off FLOPs computation
        compute_flops = False

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
                           f"metrics_{dataset_name}_{img_name}.json"), "w") as f:
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    for ax in axes.ravel():
        ax.axis("off")
    
    axes.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
    axes.ravel()[0].title.set_text("Original")
        
    for i, (name, rec) in enumerate(reconstructions.items()):
        axes.ravel()[i + 1].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
        axes.ravel()[i + 1].title.set_text(name)

    fig.tight_layout()

    plt.savefig(os.path.join(output_folder,
                             f"networks_{dataset_name}_{img_name}.png"))
    plt.close()

    # Compare pre-trained networks
    # n_rows = math.ceil((len(reconstructions) + 1) / 3)
    # n_cols = 3
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    # for ax in axes.ravel():
    #     ax.axis("off")
    
    # axes.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
    # axes.ravel()[0].title.set_text("Original")
        
    # for i, (name, rec) in enumerate(pretrained_reconstructions.items()):
    #     axes.ravel()[i + 1].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
    #     axes.ravel()[i + 1].title.set_text(name)

    # fig.tight_layout()

    # plt.savefig(os.path.join(output_folder,
    #                          f"pretrained_networks_{dataset_name}_{img_name}.png"))
    # plt.close()

    # Plot rate-distortion curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    brs = [m["bit-rate"] for _, m in metrics.items()]
    pretrained_brs = [m["bit-rate"] for _, m in pretrained_metrics.items()]

    psnrs = [m["psnr"] for _, m in metrics.items()]
    # axes[0].plot(brs, psnrs, "red", linestyle="--", linewidth=1, label="proposed")

    pretrained_psnrs = [m["psnr"] for _, m in pretrained_metrics.items()]
    axes[0].plot(pretrained_brs, pretrained_psnrs, "blue", linestyle="--", linewidth=1, label="pre-trained")

    msssim = [-10*np.log10(1-m["ms-ssim"]) for _, m in metrics.items()]
    # axes[1].plot(brs, msssim, "red", linestyle="--", linewidth=1, label="proposed")

    pretrained_msssim = [-10*np.log10(1-m["ms-ssim"]) for _, m in pretrained_metrics.items()]
    axes[1].plot(pretrained_brs, pretrained_msssim, "blue", linestyle="--", linewidth=1, label="pre-trained")

    for name, m in pretrained_metrics.items():
        axes[0].plot(m["bit-rate"], m["psnr"], "o", color="blue")
        axes[0].grid(True)
        axes[0].set_ylabel("PSNR [dB]")
        axes[0].set_xlabel("Bit-rate [bpp]")
        axes[0].title.set_text("PSNR comparison")

        axes[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]), "o", color="blue")
        axes[1].grid(True)
        axes[1].set_ylabel("MS-SSIM [dB]")
        axes[1].set_xlabel("Bit-rate [bpp]")
        axes[1].title.set_text("MS-SSIM (log) comparison")
    
    for name, m in metrics.items():
        axes[0].plot(m["bit-rate"], m["psnr"],
                     "s" if name == "teacher" else "o", label=name)
        axes[0].grid(True)
        axes[0].set_ylabel("PSNR [dB]")
        axes[0].set_xlabel("Bit-rate [bpp]")
        axes[0].title.set_text("PSNR comparison")

        axes[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]),
                     "s" if name == "teacher" else "o", label=name)
        axes[1].grid(True)
        axes[1].set_ylabel("MS-SSIM [dB]")
        axes[1].set_xlabel("Bit-rate [bpp]")
        axes[1].title.set_text("MS-SSIM (log) comparison")

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    fig.tight_layout()

    plt.savefig(os.path.join(output_folder,
                             f"curve_{dataset_name}_{img_name}.png"))
    plt.close()

###############################################################################
## Metrics (average) ##########################################################
###############################################################################

# Iterate over networks
for name, net in networks.items():
    # Iterate over images
    monitor.begin_window("inference")
    for img_name in dataset_imgs:
        # Load image
        img = Image.open(os.path.join(dataset_path, img_name)).convert("RGB")
        if dataset_name == "clic":
            img = img.crop((0, 0, 768, 512)) # For CLIC dataset
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        
        # Inference
        rv = net(x)
    mes = monitor.end_window("inference")

    avg_metrics[name]["energy-time"] = mes.time
    avg_metrics[name]["energy"] = mes.total_energy

# Iterate over pre-trained networks
for name, net in pretrained_networks.items():
    # Iterate over images
    monitor.begin_window("inference")
    for img_name in dataset_imgs:
        # Load image
        img = Image.open(os.path.join(dataset_path, img_name)).convert("RGB")
        if dataset_name == "clic":
            img = img.crop((0, 0, 768, 512)) # For CLIC dataset
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        
        # Inference
        rv = net(x)
    mes = monitor.end_window("inference")

    pretrained_avg_metrics[name]["energy-time"] = mes.time
    pretrained_avg_metrics[name]["energy"] = mes.total_energy

# Compute average metrics
for name in networks.keys():
    avg_metrics[name]["inference-time"] = np.average(avg_metrics[name]["inference-time"])
    avg_metrics[name]["energy"] = np.average(avg_metrics[name]["energy"])
    avg_metrics[name]["mse"] = np.average(avg_metrics[name]["mse"])
    avg_metrics[name]["psnr"] = np.average(avg_metrics[name]["psnr"])
    avg_metrics[name]["ms-ssim"] = np.average(avg_metrics[name]["ms-ssim"])
    avg_metrics[name]["bit-rate"] = np.average(avg_metrics[name]["bit-rate"])

# Compute pre-trained average metrics
for name in pretrained_networks.keys():
    pretrained_avg_metrics[name]["inference-time"] = np.average(pretrained_avg_metrics[name]["inference-time"])
    pretrained_avg_metrics[name]["energy"] = np.average(pretrained_avg_metrics[name]["energy"])
    pretrained_avg_metrics[name]["mse"] = np.average(pretrained_avg_metrics[name]["mse"])
    pretrained_avg_metrics[name]["psnr"] = np.average(pretrained_avg_metrics[name]["psnr"])
    pretrained_avg_metrics[name]["ms-ssim"] = np.average(pretrained_avg_metrics[name]["ms-ssim"])
    pretrained_avg_metrics[name]["bit-rate"] = np.average(pretrained_avg_metrics[name]["bit-rate"])

# Save average metrics
all_avg_metrics = {"proposed": avg_metrics, "pretrained": pretrained_avg_metrics}
with open(os.path.join(output_folder,
                       f"avg_metrics_{dataset_name}.json"), "w") as f:
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
                       f"avg_bd_metrics_{dataset_name}.json"), "w") as f:
    json.dump(avg_bd_metrics, f)

###############################################################################
## Plots (average) ############################################################
###############################################################################

# Plot average rate-distortion curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# axes[0].plot(brs, psnrs, "red", linestyle="--", linewidth=1, label="proposed")
axes[0].plot(pretrained_brs, pretrained_psnrs, "blue", linestyle="--", linewidth=1, label="pre-trained")

# axes[1].plot(brs, msssim, "red", linestyle="--", linewidth=1, label="proposed")
axes[1].plot(pretrained_brs, pretrained_msssim, "blue", linestyle="--", linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axes[0].plot(m["bit-rate"], m["psnr"], "o", color="blue")
    axes[0].grid(True)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Bit-rate [bpp]")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]), "o", color="blue")
    axes[1].grid(True)
    axes[1].set_ylabel("MS-SSIM [dB]")
    axes[1].set_xlabel("Bit-rate [bpp]")
    axes[1].title.set_text("MS-SSIM (log) comparison")

for name, m in avg_metrics.items():
    axes[0].plot(m["bit-rate"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axes[0].grid(True)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Bit-rate [bpp]")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]),
                 "s" if name == "teacher" else "o", label=name)
    axes[1].grid(True)
    axes[1].set_ylabel("MS-SSIM [dB]")
    axes[1].set_xlabel("Bit-rate [bpp]")
    axes[1].title.set_text("MS-SSIM (log) comparison")

axes[0].legend(loc="best")
axes[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_rd_curve_{dataset_name}.png"))
plt.close()

# Plot number of parameters and RD performance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

params = [m["params"] for _, m in avg_metrics.items()]
pretrained_params = [m["params"] for _, m in pretrained_avg_metrics.items()]

# axes[0].plot(params, psnrs, "red", linestyle="--", linewidth=1, label="proposed")
axes[0].plot(pretrained_params, pretrained_psnrs, "blue", linestyle="--", linewidth=1, label="pre-trained")

# axes[1].plot(params, brs, "red", linestyle="--", linewidth=1, label="proposed")
axes[1].plot(pretrained_params, pretrained_brs, "blue", linestyle="--", linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axes[0].plot(m["params"], m["psnr"], "o", color="blue")
    axes[0].grid(True)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Number of parameters")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["params"], m["bit-rate"], "o", color="blue")
    axes[1].grid(True)
    axes[1].set_ylabel("Bit-rate [bpp]")
    axes[1].set_xlabel("Number of parameters")
    axes[1].title.set_text("Bit-rate comparison")

for name, m in avg_metrics.items():
    axes[0].plot(m["params"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axes[0].grid(True)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Number of parameters")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["params"], m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axes[1].grid(True)
    axes[1].set_ylabel("Bit-rate [bpp]")
    axes[1].set_xlabel("Number of parameters")
    axes[1].title.set_text("Bit-rate comparison")

axes[0].legend(loc="best")
axes[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_param_curve_{dataset_name}.png"))
plt.close()

# Plot inference time and RD performance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

times = [m["inference-time"] for _, m in avg_metrics.items()]
pretrained_times = [m["inference-time"] for _, m in pretrained_avg_metrics.items()]

# axes[0].plot(params, psnrs, "red", linestyle="--", linewidth=1, label="proposed")
axes[0].plot(pretrained_times, pretrained_psnrs, "blue", linestyle="--", linewidth=1, label="pre-trained")

# axes[1].plot(params, brs, "red", linestyle="--", linewidth=1, label="proposed")
axes[1].plot(pretrained_times, pretrained_brs, "blue", linestyle="--", linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axes[0].plot(m["inference-time"], m["psnr"], "o", color="blue")
    axes[0].grid(True)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Inference time [s]")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["inference-time"], m["bit-rate"], "o", color="blue")
    axes[1].grid(True)
    axes[1].set_ylabel("Bit-rate [bpp]")
    axes[1].set_xlabel("Inference time [s]")
    axes[1].title.set_text("Bit-rate comparison")

for name, m in avg_metrics.items():
    axes[0].plot(m["inference-time"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axes[0].grid(True)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Inference time [s]")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["inference-time"], m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axes[1].grid(True)
    axes[1].set_ylabel("Bit-rate [bpp]")
    axes[1].set_xlabel("Inference time [s]")
    axes[1].title.set_text("Bit-rate comparison")

axes[0].legend(loc="best")
axes[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_time_curve_{dataset_name}.png"))
plt.close()

# Plot energy and RD performance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

energies = [m["energy"] for _, m in avg_metrics.items()]
pretrained_energies = [m["energy"] for _, m in pretrained_avg_metrics.items()]

# axes[0].plot(params, psnrs, "red", linestyle="--", linewidth=1, label="proposed")
axes[0].plot(pretrained_energies, pretrained_psnrs, "blue", linestyle="--", linewidth=1, label="pre-trained")

# axes[1].plot(params, brs, "red", linestyle="--", linewidth=1, label="proposed")
axes[1].plot(pretrained_energies, pretrained_brs, "blue", linestyle="--", linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axes[0].plot(m["energy"], m["psnr"], "o", color="blue")
    axes[0].grid(True)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Energy [J]")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["energy"], m["bit-rate"], "o", color="blue")
    axes[1].grid(True)
    axes[1].set_ylabel("Bit-rate [bpp]")
    axes[1].set_xlabel("Energy [J]")
    axes[1].title.set_text("Bit-rate comparison")

for name, m in avg_metrics.items():
    axes[0].plot(m["energy"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axes[0].grid(True)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Energy [J]")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["energy"], m["bit-rate"],
                 "s" if name == "teacher" else "o", label=name)
    axes[1].grid(True)
    axes[1].set_ylabel("Bit-rate [bpp]")
    axes[1].set_xlabel("Energy [J]")
    axes[1].title.set_text("Bit-rate comparison")

axes[0].legend(loc="best")
axes[1].legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_energy_curve_{dataset_name}.png"))
plt.close()

# Plot mse and channel
fig, axes = plt.subplots(1, 1, figsize=(6, 5))

axes.plot(Ns[1:], [m["mse"] for _, m in avg_metrics.items()][1:], "red", linestyle="--", linewidth=1)
for i, (name, m) in enumerate(avg_metrics.items()):
    axes.plot(Ns[i], m["mse"],
              "s" if name == "teacher" else "o", label=name)
axes.grid(True)
axes.set_ylabel("MSE")
axes.set_xlabel("Number of channels")
axes.title.set_text("MSE comparison")
axes.legend(loc="best")

fig.tight_layout()

plt.savefig(os.path.join(output_folder,
                         f"avg_mse_curve_{dataset_name}.png"))
plt.close()
import os
from datetime import datetime
import json

import math
import numpy as np
import scipy.interpolate

import torch
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

import matplotlib.pyplot as plt
from PIL import Image

from models import ScaleHyperprior

plt.rcParams["axes.prop_cycle"] = plt.rcParams["axes.prop_cycle"][1:]

device = "cuda" if torch.cuda.is_available() else "cpu"
metric = "mse"

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


# Create output directory
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"/home/ids/fallemand-24/PRIM/kd_lic_experiments/test_res/{time_stamp}"
os.makedirs(output_folder)

###############################################################################
## Networks ###################################################################
###############################################################################

# Load networks
M = 192
Ns = [128, 16, 32, 64, 96]
ids = [None, 258263, 258258, 258259, 258262]

networks = {
    "teacher": None,
    "student_16": None,
    "student_32": None,
    "student_64": None,
    "student_96": None,
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
            "psnr": [],
            "ms-ssim": [],
            "bit-rate": [],
        }

###############################################################################
## Inference, Metrics and Plots ###############################################
###############################################################################

# Find images
dataset_path = "/home/ids/fallemand-24/PRIM/data/kodak"
# dataset_path = "/home/ids/fallemand-24/PRIM/data/clic/clic_validation"
# dataset_path = "/home/ids/fallemand-24/PRIM/data/clic/clic_test"
dataset_name = dataset_path.split("/")[-1]

dataset_imgs = [p for p in os.listdir(dataset_path) if p.endswith(".png")]

for img_name in dataset_imgs:
    # Load image
    img = Image.open(os.path.join(dataset_path,
                                               img_name)).convert("RGB")
    # img = img.crop((0, 0, 768, 512)) # For CLIC dataset
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    img_name = img_name.split(".")[0]

    # Inference with networks
    outputs = {}
    with torch.no_grad():
        for name, net in networks.items():
            rv = net(x)
            rv["x_hat"].clamp_(0, 1)
            outputs[name] = rv

    reconstructions = {name: transforms.ToPILImage()(out["x_hat"].squeeze())
                       for name, out in outputs.items()}

    diffs = [torch.mean((out["x_hat"] - x).abs(), axis=1).squeeze()
             for out in outputs.values()]

    # Inference with pre-trained networks
    pretrained_outputs = {}
    with torch.no_grad():
        for name, net in pretrained_networks.items():
            rv = net(x)
            rv["x_hat"].clamp_(0, 1)
            pretrained_outputs[name] = rv

    pretrained_reconstructions = {name: transforms.ToPILImage()(out["x_hat"].squeeze())
                                  for name, out in pretrained_outputs.items()}

    pretrained_diffs = [torch.mean((out["x_hat"] - x).abs(), axis=1).squeeze()
                        for out in pretrained_outputs.values()]
    
    # Compare networks
    fix, axes = plt.subplots((len(reconstructions) + 2) // 3, 3,
                             figsize=(16, 12))
    for ax in axes.ravel():
        ax.axis("off")
    
    axes.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
    axes.ravel()[0].title.set_text("Original")
        
    for i, (name, rec) in enumerate(reconstructions.items()):
        axes.ravel()[i + 1].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
        axes.ravel()[i + 1].title.set_text(name)

    plt.savefig(os.path.join(output_folder,
                             f"networks_{dataset_name}_{img_name}.png"))
    plt.close()

    # Compare pre-trained networks
    # fix, axes = plt.subplots((len(pretrained_reconstructions) + 2) // 3, 3,
    #                          figsize=(16, 12))
    # for ax in axes.ravel():
    #     ax.axis("off")
    
    # axes.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
    # axes.ravel()[0].title.set_text("Original")
        
    # for i, (name, rec) in enumerate(pretrained_reconstructions.items()):
    #     axes.ravel()[i + 1].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
    #     axes.ravel()[i + 1].title.set_text(name)

    # plt.savefig(os.path.join(output_folder,
    #                          f"pretrained_networks_{dataset_name}_{img_name}.png"))
    # plt.close()

    # Compute metrics for networks
    metrics = {}
    for name, out in outputs.items():
        metrics[name] = {
            "psnr": compute_psnr(x, out["x_hat"]),
            "ms-ssim": compute_msssim(x, out["x_hat"]),
            "bit-rate": compute_bpp(out),
        }

        avg_metrics[name]["psnr"].append(metrics[name]["psnr"])
        avg_metrics[name]["ms-ssim"].append(metrics[name]["ms-ssim"])
        avg_metrics[name]["bit-rate"].append(metrics[name]["bit-rate"])

    # Compute metrics for pre-trained networks
    pretrained_metrics = {}
    for name, out in pretrained_outputs.items():
        pretrained_metrics[name] = {
            "psnr": compute_psnr(x, out["x_hat"]),
            "ms-ssim": compute_msssim(x, out["x_hat"]),
            "bit-rate": compute_bpp(out),
        }

        pretrained_avg_metrics[name]["psnr"].append(pretrained_metrics[name]["psnr"])
        pretrained_avg_metrics[name]["ms-ssim"].append(pretrained_metrics[name]["ms-ssim"])
        pretrained_avg_metrics[name]["bit-rate"].append(pretrained_metrics[name]["bit-rate"])

    # Save metrics
    all_metrics = metrics | pretrained_metrics
    with open(os.path.join(output_folder,
                           f"metrics_{dataset_name}_{img_name}.json"), "w") as f:
        json.dump(all_metrics, f)

    # Plot rate-distortion curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.figtext(.5, 0., "(upper-left is better)", fontsize=12, ha="center")

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
        axes[0].grid()
        axes[0].set_ylabel("PSNR [dB]")
        axes[0].set_xlabel("Bit-rate [bpp]")
        axes[0].title.set_text("PSNR comparison")

        axes[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]), "o", color="blue")
        axes[1].grid()
        axes[1].set_ylabel("MS-SSIM [dB]")
        axes[1].set_xlabel("Bit-rate [bpp]")
        axes[1].title.set_text("MS-SSIM (log) comparison")
    
    for name, m in metrics.items():
        axes[0].plot(m["bit-rate"], m["psnr"],
                     "s" if name == "teacher" else "o", label=name)
        axes[0].grid()
        axes[0].set_ylabel("PSNR [dB]")
        axes[0].set_xlabel("Bit-rate [bpp]")
        axes[0].title.set_text("PSNR comparison")

        axes[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]),
                     "s" if name == "teacher" else "o", label=name)
        axes[1].grid()
        axes[1].set_ylabel("MS-SSIM [dB]")
        axes[1].set_xlabel("Bit-rate [bpp]")
        axes[1].title.set_text("MS-SSIM (log) comparison")

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    plt.savefig(os.path.join(output_folder,
                             f"curve_{dataset_name}_{img_name}.png"))
    plt.close()

###############################################################################
## Average Metrics and Plots ##################################################
###############################################################################

# Compute average metrics
for name in networks.keys():
    avg_metrics[name]["psnr"] = np.average(avg_metrics[name]["psnr"])
    avg_metrics[name]["ms-ssim"] = np.average(avg_metrics[name]["ms-ssim"])
    avg_metrics[name]["bit-rate"] = np.average(avg_metrics[name]["bit-rate"])

# Compute pre-trained average metrics
for name in pretrained_networks.keys():
    pretrained_avg_metrics[name]["psnr"] = np.average(pretrained_avg_metrics[name]["psnr"])
    pretrained_avg_metrics[name]["ms-ssim"] = np.average(pretrained_avg_metrics[name]["ms-ssim"])
    pretrained_avg_metrics[name]["bit-rate"] = np.average(pretrained_avg_metrics[name]["bit-rate"])

# Save average metrics
all_avg_metrics = {"proposed": avg_metrics, "pretrained": pretrained_avg_metrics}
with open(os.path.join(output_folder,
                       f"avg_metrics_{dataset_name}.json"), "w") as f:
    json.dump(all_metrics, f)

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

# Plot average rate-distortion curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plt.figtext(.5, 0., "(upper-left is better)", fontsize=12, ha="center")

# axes[0].plot(brs, psnrs, "red", linestyle="--", linewidth=1, label="proposed")
axes[0].plot(pretrained_brs, pretrained_psnrs, "blue", linestyle="--", linewidth=1, label="pre-trained")

# axes[1].plot(brs, msssim, "red", linestyle="--", linewidth=1, label="proposed")
axes[1].plot(pretrained_brs, pretrained_msssim, "blue", linestyle="--", linewidth=1, label="pre-trained")

for name, m in pretrained_avg_metrics.items():
    axes[0].plot(m["bit-rate"], m["psnr"], "o", color="blue")
    axes[0].grid()
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Bit-rate [bpp]")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]), "o", color="blue")
    axes[1].grid()
    axes[1].set_ylabel("MS-SSIM [dB]")
    axes[1].set_xlabel("Bit-rate [bpp]")
    axes[1].title.set_text("MS-SSIM (log) comparison")

for name, m in avg_metrics.items():
    axes[0].plot(m["bit-rate"], m["psnr"],
                 "s" if name == "teacher" else "o", label=name)
    axes[0].grid()
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xlabel("Bit-rate [bpp]")
    axes[0].title.set_text("PSNR comparison")

    axes[1].plot(m["bit-rate"], -10*np.log10(1-m["ms-ssim"]),
                 "s" if name == "teacher" else "o", label=name)
    axes[1].grid()
    axes[1].set_ylabel("MS-SSIM [dB]")
    axes[1].set_xlabel("Bit-rate [bpp]")
    axes[1].title.set_text("MS-SSIM (log) comparison")

axes[0].legend(loc="best")
axes[1].legend(loc="best")

plt.savefig(os.path.join(output_folder,
                         f"avg_curve_{dataset_name}.png"))
plt.close()
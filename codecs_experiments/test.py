import os
import io
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

from zeus.monitor import ZeusMonitor
# from zeus.device.cpu import get_current_cpu_index
import pynvml

import matplotlib.pyplot as plt
from PIL import Image

# Set-up matplotlib
plt.rcParams["axes.prop_cycle"] = plt.rcParams["axes.prop_cycle"][1:]

# Create output directory
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"/home/ids/fallemand-24/PRIM/codecs_experiments/test_res/{time_stamp}"
os.makedirs(output_folder)

###############################################################################
## Utils ######################################################################
###############################################################################

def pillow_encode(img, fmt="jpeg", quality=10):
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    return rec, bpp


def find_closest_bpp(target, img, fmt="jpeg"):
    lower = 0
    upper = 100
    prev_mid = upper

    def _psnr(a, b):
        a = np.asarray(a).astype(np.float32)
        b = np.asarray(b).astype(np.float32)
        mse = np.mean(np.square(a - b))
        return 20*math.log10(255.) -10. * math.log10(mse + 10e-8)
    
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    psnr_val = _psnr(rec, img)
    return rec, bpp, psnr_val

def find_closest_psnr(target, img, fmt="jpeg"):
    lower = 0
    upper = 100
    prev_mid = upper
    
    def _psnr(a, b):
        a = np.asarray(a).astype(np.float32)
        b = np.asarray(b).astype(np.float32)
        mse = np.mean(np.square(a - b))
        return 20*math.log10(255.) -10. * math.log10(mse + 10e-8)
    
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        prev_mid = mid
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        psnr_val = _psnr(rec, img)
        if psnr_val > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp, psnr_val

def find_closest_msssim(target, img, fmt="jpeg"):
    lower = 0
    upper = 100
    prev_mid = upper
    
    def _mssim(a, b):
        a = torch.from_numpy(np.asarray(a).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        b = torch.from_numpy(np.asarray(b).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return ms_ssim(a, b, data_range=255.).item()

    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        prev_mid = mid
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        msssim_val = _mssim(rec, img)
        if msssim_val > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp, msssim_val


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse + 10e-8)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


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
## Codecs #####################################################################
###############################################################################

# List of codecs
codecs = ["jpeg", "jpeg2000", "webp", "png"]

# Create dict for average metrics
avg_metrics = {}
for codec in codecs:
    avg_metrics[codec] = {
            "inference-time": [],
            "zeus-time": [],
            "zeus-fps": [],
            "zeus-energy": [],
            "pynvml-time": [],
            "pynvml-fps": [],
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

# Dataset codec
dataset_name = DATASET_PATH.split("/")[-1]

# Images path
dataset_imgs = [p for p in os.listdir(DATASET_PATH) if p.endswith(".png")]

###############################################################################
## Inference and Metrics (single image) #######################################
###############################################################################

# Criterion
criterion = nn.MSELoss()

# Iterate over images
for img_name in dataset_imgs:
    # Load image
    img = Image.open(os.path.join(DATASET_PATH, img_name)).convert("RGB")
    if dataset_name == "clic":
        img = img.crop((0, 0, 768, 512)) # For CLIC dataset
    x = transforms.ToTensor()(img).unsqueeze(0)
    img_name = img_name.split(".")[0]

    # Compression and metrics for codecs
    outputs = {}
    metrics = {}
    with torch.no_grad():
        for codec in codecs:
            # Compression
            start = time.time()
            out, bpp = pillow_encode(img, codec)
            stop = time.time()

            # Save output
            outputs[codec] = out

            # Compute metrics
            metrics[codec] = {
                "inference-time": stop - start,
                "mse": criterion(transforms.ToTensor()(out).unsqueeze(0), x).item(),
                "psnr": compute_psnr(transforms.ToTensor()(out).unsqueeze(0), x),
                "ms-ssim": compute_msssim(transforms.ToTensor()(out).unsqueeze(0), x),
                "bit-rate": bpp,
            }
            avg_metrics[codec]["inference-time"].append(metrics[codec]["inference-time"])
            avg_metrics[codec]["mse"].append(metrics[codec]["mse"])
            avg_metrics[codec]["psnr"].append(metrics[codec]["psnr"])
            avg_metrics[codec]["ms-ssim"].append(metrics[codec]["ms-ssim"])
            avg_metrics[codec]["bit-rate"].append(metrics[codec]["bit-rate"])

    # Save metrics
    # all_metrics = metrics | pretrained_metrics
    # with open(os.path.join(output_folder,
    #                        f"metrics_{dataset_name}_{img_name}.json"),
    #                        "w", encoding="utf-8") as f:
    #     json.dump(all_metrics, f)

###############################################################################
## Plots (single image) #######################################################
###############################################################################

    reconstructions = {codec: out
                       for codec, out in outputs.items()}

    diffs = [torch.mean((transforms.ToTensor()(out).unsqueeze(0) - x).abs(), axis=1).squeeze()
             for out in outputs.values()]

    # Compare codecs
    n_rows = math.ceil((len(reconstructions) + 1) / 3)
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    for ax in axs.ravel():
        ax.axis("off")

    axs.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
    axs.ravel()[0].title.set_text("Original")

    for i, (codec, out) in enumerate(reconstructions.items()):
        axs.ravel()[i + 1].imshow(out.crop((468, 212, 768, 512))) # cropped for easy comparison
        axs.ravel()[i + 1].title.set_text(codec)

    fig.tight_layout()

    plt.savefig(os.path.join(output_folder,
                             f"codecs_{dataset_name}_{img_name}.png"))
    plt.close()

###############################################################################
## Metrics (average) ##########################################################
###############################################################################

DATASET_ITER = 50

loaded_dataset_imgs = []
for img_codec in dataset_imgs:
    # Load image
    img = Image.open(os.path.join(DATASET_PATH, img_codec)).convert("RGB")
    if dataset_name == "clic":
        img = img.crop((0, 0, 768, 512)) # For CLIC dataset
    loaded_dataset_imgs.append(img)

# Zeus energy consumption
# zeus_monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
zeus_monitor = ZeusMonitor(approx_instant_energy=False)

# Iterate over codecs
for codec in codecs:
    # Iterate over images
    zeus_monitor.begin_window("compression")
    for _ in range(DATASET_ITER):
        for x in loaded_dataset_imgs:
            # Compression
            pillow_encode(img, codec)
    mes = zeus_monitor.end_window("compression")

    sec_per_frame = (mes.time / (DATASET_ITER * len(loaded_dataset_imgs)))
    avg_metrics[codec]["zeus-time"] = 1000 * sec_per_frame # Convert to ms/frame
    avg_metrics[codec]["zeus-fps"] = 1 / sec_per_frame # Convert to FPS
    avg_metrics[codec]["zeus-energy"] = (1000 * mes.total_energy) / (DATASET_ITER * len(loaded_dataset_imgs)) # Convert to mJ/frame

# pynvml energy consumption
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # First GPU

# Iterate over networks
for codec in codecs:
    start_time = time.time()
    start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    # Iterate over images
    for _ in range(DATASET_ITER):
        for x in loaded_dataset_imgs:
            # Compression
            pillow_encode(img, codec)
    torch.cuda.synchronize()  # Synchronizes CPU and GPU time.
    elapsed_time = time.time() - start_time
    consumed_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy

    avg_metrics[codec]["pynvml-time"] = (1000 * elapsed_time) / (DATASET_ITER * len(loaded_dataset_imgs)) # Convert to ms/frame
    avg_metrics[codec]["pynvml-fps"] = 1 / avg_metrics[codec]["pynvml-time"] # Convert to FPS
    avg_metrics[codec]["pynvml-energy"] = consumed_energy / (DATASET_ITER * len(loaded_dataset_imgs)) # Convert to mJ/frame

# Compute average metrics
for codec in codecs:
    avg_metrics[codec]["inference-time"] = np.average(avg_metrics[codec]["inference-time"])
    avg_metrics[codec]["mse"] = np.average(avg_metrics[codec]["mse"])
    avg_metrics[codec]["psnr"] = np.average(avg_metrics[codec]["psnr"])
    avg_metrics[codec]["ms-ssim"] = np.average(avg_metrics[codec]["ms-ssim"])
    avg_metrics[codec]["bit-rate"] = np.average(avg_metrics[codec]["bit-rate"])

# Save average metrics
with open(os.path.join(output_folder,
                       f"avg_metrics_{dataset_name}.json"),
                       "w", encoding="utf-8") as f:
    json.dump(avg_metrics, f)
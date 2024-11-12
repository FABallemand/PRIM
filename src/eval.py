import torch 
import os 
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from step import AverageMeter  # TODO change import
import math 
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import json
import argparse
from dataset import TestKodakDataset # TODO change import
from torch.utils.data import DataLoader
from os.path import join 

# from training.loss import RateDistortionLoss

import sys
import random
import numpy as np

import bd_metrics # TODO change import

from compressai.zoo import cheng2020_attn, mbt2018_mean, bmshj2018_hyperprior, bmshj2018_factorized_relu, bmshj2018_factorized, mbt2018


import json


Colors = {
    "Cheng2020": ["g",'*'],
    "Minnen2018": ["m",'o'],
    "Balle2018": ["c",'.'],
    "Balle_relu":["g",'+'],
    "Balle_gdn":["b",'x'], 
    "mbt2018_JA":["w",'-'],
    }
Colors_vs_base = Colors

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def set_seed(seed=66):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def load_models(models_path,model_checkpoint, device, model_type):
    # TODO load your pretrained models as you prefer!
    
    res = {}
    for model_check in model_checkpoint:
    
        model_path = join(models_path, model_check)
        checkpoint = torch.load(model_path, map_location=device)

        name = (model_path.split(os.sep)[-1]).replace('.pth.tar','')

        if(model_type == 'tcm'):
            print('Loading TCM')
            model = models[model_type](config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
            model = model.to(device)
            model.eval()
            dictory = {}
            for k, v in checkpoint["state_dict"].items():
                dictory[k.replace("module.", "")] = v
            model.load_state_dict(dictory)
            model.update(force = True)

        lambd = 0.0
        if 'q1' in name:
            lambd = 0.0018
        elif 'q2' in name:
            lambd = 0.0035
        elif 'q3' in name:
            lambd = 0.0067
        elif 'q4' in name:
            lambd = 0.0130
        elif 'q5' in name:
            lambd = 0.0250
        elif 'q6' in name:
            lambd = 0.0483
        else:
            raise NotImplementedError(f'{name} does not contains qx')

        res[name] = {
            "model": model,
            "psnr": AverageMeter(),
            "ms_ssim": AverageMeter(),
            "bpps": AverageMeter(),
            "rate": AverageMeter(),
            "criterion": None, # RateDistortionLoss(lmbda=lambd),
            "loss": AverageMeter()
            }
        print(f'{model_path} loaded')
    print()
    return res


def read_image(filepath):
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


@torch.no_grad()
def inference_entropy_estimation(model,x, x_padded, padding,criterion= None):

    out  = model(x_padded)
    if criterion is not None:
        out_criterion = criterion(out, x_padded)
        loss = out_criterion["loss"].item()
    else:
        loss = 0.0
    
    # out["x_hat"] = F.pad(out["x_hat"], unpad)
    out["x_hat"] = crop(out["x_hat"], padding)

    metrics = compute_metrics(x, out["x_hat"], 255)
    size = out['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]

    y_bpp = torch.log(out["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
                        
    bpp = y_bpp
    rate = bpp.item()*num_pixels 

    return metrics, bpp, rate, out["x_hat"], loss


@torch.no_grad()
def inference(model,x, x_padded, padding):

    out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    rate = bpp*num_pixels 

    return metrics, torch.tensor([bpp]), rate, out_dec["x_hat"]


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


@torch.no_grad()
def eval_models(models, dataloader, device, dataset = 'HD'):

    
    print("Starting inferences")

    res_metrics = {}
    for j,(x,_) in enumerate(dataloader):

        print(j)
        x = x.to(device)

        x_padded, padding = pad(x, 128)

        if(j==0):
            print(f'unpad: {x.shape}')
            print(f'input: {x_padded.shape}')


        if(x.shape != x_padded.shape):
            print(f'{x.shape} --> {x_padded.shape}')


        for model_type in list(models.keys()):
            for qp in sorted(list(models[model_type].keys())):
                model = models[model_type][qp]['model']

                if True:    
                    metrics, bpp, rate, x_hat = inference(model,x,x_padded, padding)
                    loss = 0.0
                else:
                    metrics, bpp, rate, x_hat, loss = inference_entropy_estimation(
                        model,
                        x,
                        x_padded,
                        padding)

                    
                models[model_type][qp]['psnr'].update(metrics["psnr"])
                models[model_type][qp]['ms_ssim'].update(metrics["ms-ssim"])
                models[model_type][qp]['bpps'].update(bpp.item())
                models[model_type][qp]['rate'].update(rate)
                models[model_type][qp]['loss'].update(loss)

    for model_type in list(models.keys()):
        model_res = {}
        print(model_type)
        for qp in list(models[model_type].keys()):
            qp_name = str(qp).split('-')[0]
            model_res[qp_name] = {
                'psnr': models[model_type][qp]['psnr'].avg,
                'mssim': models[model_type][qp]['ms_ssim'].avg,
                'bpp': models[model_type][qp]['bpps'].avg,
                'rate': models[model_type][qp]['rate'].avg,
                'loss': models[model_type][qp]['loss'].avg
            }
            print(f'{qp}: {model_res[qp_name]}')
        res_metrics[model_type] = model_res

    return res_metrics   


def extract_specific_model_performance(metrics, type):

    nms = list(metrics[type].keys())

    psnr = []
    mssim = []
    bpp = []
    rate = []
    for names in nms:
        psnr.append(metrics[type][names]["psnr"])
        mssim.append(metrics[type][names]["mssim"])
        bpp.append(metrics[type][names]["bpp"])
        rate.append(metrics[type][names]["rate"])

    
    return sorted(psnr), sorted(mssim), sorted(bpp), sorted(rate)


def plot_rate_distorsion_psnr(metrics, savepath, colors = Colors):

    print(f'plotting on {savepath}')

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
    for type_name in metrics.keys():

        psnr, mssim, bpp, rate = extract_specific_model_performance(metrics, type_name)      
        cols = colors[type_name]      
        axes.plot(bpp, psnr,cols[1],color = cols[0], label = type_name)
        axes.plot(bpp, psnr,'-',color = cols[0])
        axes.plot(bpp, psnr,cols[1],color =  cols[0])


    axes.set_ylabel('PSNR [dB]')
    axes.set_xlabel('Bit-rate [bpp]')
    axes.title.set_text(f'PSNR comparison')
    axes.grid()
    axes.legend(loc='best')

    # for ax in axes:
    axes.grid(True)
    plt.savefig(savepath)
    plt.close()      


def produce_metrics(configs, dataset = 'HD'):

   # Loading dict of models
    models = {}

    device = "cpu"
    for model_arch in configs:
        res = {}

        for qp in range(1,7):
            if model_arch == 'Cheng2020':
                net = cheng2020_attn(quality=qp, pretrained=True).eval().to(device)
            elif model_arch == 'Minnen2018':
                net = mbt2018_mean(quality=qp, pretrained=True).eval().to(device)
            elif model_arch == 'Balle2018':
                net = bmshj2018_hyperprior(quality=qp, pretrained=True).eval().to(device)
            elif model_arch== 'Balle_relu':
                net=  bmshj2018_factorized_relu(quality=qp, pretrained=True).eval().to(device)  
            elif model_arch== 'Balle_gdn':
                net=  bmshj2018_factorized(quality=qp, pretrained=True).eval().to(device)  
            elif model_arch=='mbt2018_JA':
                net=mbt2018(quality=qp,pretrained=True).eval().to(device)
            
            res[f'q{qp}-model'] = {   
                "model": net,
                "psnr": AverageMeter(),
                "ms_ssim": AverageMeter(),
                "bpps": AverageMeter(),
                "rate": AverageMeter(),
                "criterion": None,
                "loss": AverageMeter()
            }
        models[model_arch] = res

 # Use the provided dataset path, default to 'active folder/dataset/kodak'
    #if dataset_path is None:
    dataset_path = os.path.join(os.getcwd(), 'dataset', 'HD')
    images_path = dataset_path

    test_dataset = TestKodakDataset(data_dir=images_path, image_size=-1, crop=False, random_crop=False, get_img_name=True)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)  
    metrics = eval_models(models, test_dataloader, device, dataset=dataset)
    return metrics


def produce_bd_metrics(metrics, baseline_name = 'local_attention', save_file = '/save/dir'):
    
    if(baseline_name not in metrics.keys()):
       print(f'{baseline_name} not found')
       sys.exit(1)

    psnr_base, _, _, rate_base = extract_specific_model_performance(metrics, 'local_attention')

    for type_name in metrics.keys():
        if(type_name == 'local_attention'):
            continue

        print(type_name)
        psnr, _, _, rate = extract_specific_model_performance(metrics, type_name)   
        print(f'DB-PSNR: {BD_PSNR(rate_base, psnr_base, rate, psnr)}')
        print(f'DB-RATE: {BD_RATE(rate_base, psnr_base, rate, psnr)}')

        with open(save_file, 'a') as f:
            f.write(f'## {type_name}\n')
            f.write(f'DB-PSNR: {BD_PSNR(rate_base, psnr_base, rate, psnr)} <br>\n')
            f.write(f'DB-RATE: {BD_RATE(rate_base, psnr_base, rate, psnr)} <br>\n\n')

        print('---------\n')


if __name__ == "__main__":

    set_seed()
    
    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    
    my_parser.add_argument("--metrics", default="none", type=str, help='metrics json file')
    
    my_parser.add_argument("--dataset", default="HD", type=str)

    args = my_parser.parse_args()
    
    configs = [
        #"Cheng2020",
        "Minnen2018",
        "Balle2018",
        "Balle_gdn",
        "mbt2018_JA",
    ]

    new_metrics = {}
    if(args.metrics == "none"):

        # print(config)
        new_metrics = produce_metrics(
            configs,
            dataset = args.dataset)
        dataset_path = os.path.join(os.getcwd(), "dataset", "HD")
        save_path = dataset_path
    
        # os.makedirs(save_path, exist_ok=True)

        print(f"Results will be saved on {save_path}")

        file_path = join(save_path, f"res.json")
        with open(file_path, "w") as outfile:
            json.dump(new_metrics, outfile)

        save_path_img = join(save_path, f"res.pdf")

        colors = Colors
    else:
        work_path = "/".join(args.metrics.split("/")[:-1])

        with open(args.metrics) as json_file:
            new_metrics = json.load(json_file)
        save_path_img = join(work_path,f"res.pdf")
        save_path = work_path

        colors = Colors_vs_base

    plot_rate_distorsion_psnr(new_metrics, save_path_img, colors=colors)

    # produce_bd_metrics(new_metrics, save_file=join(save_path,f'bd_res.txt'))
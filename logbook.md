# PRIM Logbook

## ??/09/2024
- Read articles

## ??/09/2024
- Meeting

## ??/09/2024
- Read articles

## 10/10/2024
- Quick meeting

## 16/10/2024
- Kickoff meeting

## 17/10/2024
- First connexion to GPU cluster

## 18/10/2024
- Reproduce Ballé results:
  - Test compressAI manually (index error)
  - Use compressAI example notebooks (dimension error)
    - Image size multiple of 64 (https://github.com/InterDigitalInc/CompressAI/issues/252)
    - Error depends of the model...

## 19/10/2024
- Reproduce Ballé results:
  - Connect to GPU cluster
> ssh -Y fallemand-24@gpu-gw.enst.fr
  - Find INFRES storage (uid: 87287 -> ir700)
  - Connect to interactive mode
  - Install conda
  - Create balle_reproduction env
  - Clone PRIM GitHub repository
  - Download vimeo dataset

## 20/10/2024
- Reproduce Ballé results:
  - Modify train scripts (to work with vimeo dataset)
  - Install pytorch on balle_reproduction env
> conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  - Install pip on balle_reproduction env
> pip install pip
  - Install compressai on balle_reproduction env
> pip install compressai
  - Connect to GPU cluster with VS Code
  - Submit script to the queue
  - Fix small issues and test things
  - First training (224584)
  - Explore results in compressai notebooks

## 21/10/2024
- Reproduce Ballé results:
  - More training from checkpoint of last training (224844)
- LIC understanding:
  - Multiple decoders architecture to better handle textures and details (https://www.youtube.com/watch?v=DwZBlGy1lbg&list=PL_bDvITUYucAh0c2I_jR2oImaz9Yzl4IS&ab_channel=ComputerVisionFoundationVideos)
  - Stanford course on LIC (https://www.youtube.com/watch?v=H7dvh35xNuE&ab_channel=StanfordOnline)
    - Trick to back pro through quantization
    - 39:00 ??
  - Need to understand entropy coding...

## 22/10/2024
- Datasets:
  - Download CLIC dataset
  - Install tensorflow datasets to access Open Images dataset
  - Create script to download all datasets
- Model evaluation:
  - Code understanding compressai/provided files
  - Write eval script

## 23/10/2024
- Model evaluation:
  - Evaluate models with compressai notebooks
  - Evaluate models with script

## 29/10/2024
- Reproduce Ballé results:
  - More training from checkpoint of last training (225698)

## 05/11/2024
- Reproduce Ballé results:
  - More training from checkpoint of last training (227892)

## 12/11/2024
- Meeting:
  - Find lambda values used in papers
  - Do training for each lambda value for the same amount of epochs (idealy the same amount of the pre-trained model)
  - Visualise Rate distortion curves (distortion in function of lambda like in papers)
  - Compute BDPSNR / BD Rate to compare models (curves)
- BDPSNR curve:
  - Create train/test scripts
  - Start first 3 training

## 12/11/2024
- BDPSNR curve:
  - Start last 5 training

## 22/11/2024
- Meeting:
  - Discuss results
  - Need to evaluate on kodak dataset
- Fix issue with pre-trained models (pretrained=True)

## 24/11/2024
- BDPSNR curve:
  - Create script to evaluate on kodak dataset

## 25/11/2024
- Write intermediary report

## 26/11/2024
- BDPSNR curve:
  - Find issue related to `quality`
  - Retrain models

## 29/11/2024
- Write intermediary report

## 2/11/2024
- Compute BD-rate and BD-PSNR:
  - https://github.com/Anserw/Bjontegaard_metric/tree/master
  - Do evaluation on CLIC dataset

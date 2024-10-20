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
  - First training
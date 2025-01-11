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

## 2/12/2024
- Compute BD-rate and BD-PSNR:
  - https://github.com/Anserw/Bjontegaard_metric/tree/master
  - Do evaluation on CLIC dataset

## 4/12/2024
- Meeting:
  - `quality` 8 is too much for our project, skip it to avoid useless computation
  - Understand knowledge distillation (KD)
    - Papers (https://arxiv.org/abs/1503.02531, https://arxiv.org/pdf/2309.02529)
    - PyTorch tutorial (https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)

## 9/12/2024
- KD PyTorch tutorial reading:
  - 3 KD methods
    - Distillatio loss on logits
    - Cosine similarity on flattened hidden representation
    - MSE on hidden representation
- Understand KD:
  - Read Distilling the Knowledge in a Neural Network
    - Beginning easy to understand

## 11/12/2024
- Meeting:
  - Understand KD
    - Papers (https://arxiv.org/pdf/2201.02624, https://arxiv.org/pdf/2309.02529)
  - Apply KD to simple AE
    - Try different ways
      - Loss on latent (cosine similarity, KL divergeance)
      - Loss on reconstruction (MSE, MS-SSIM)
  - Apply KD to LIC
    - On decoder part (image + hyperprior decoder)
- Apply KD to AE:
  - Create models
  - Create script to train teacher AE

## 12/12/2024
- Apply KD to AE:
  - Train teacher AE
  - Create script to train student AE
- Understand KD:
  - Read and create reading notes
  - Read Distilling the Knowledge in a Neural Network
  - Read Microdosing: Knowledge Distillation for GAN based Compression

## 13/12/2024
- Apply KD to AE:
  - Test teacher AE
  - Bad results
  - Noise scaled by 0.2 too high -> change for 0.1??
  - Train student AE
  - Train student AE with KD
- Understand KD:
  - Read and create reading notes
  - Read Microdosing: Knowledge Distillation for GAN based Compression

## 14/12/2024
- Understand KD:
  - Read and create reading notes
  - Read Fast and High-Performance Learned Image Compression With Improved Checkerboard Context Model, Deformable Residual Module, and Knowledge Distillation
  - Read RD efficient FPGA implementation of LIC model without complex hardware space design exploration (DRAFT)

## 18/12/2024
- Meeting:
  - Apply KD to AE
    - Difficult to analyse results when teacher does not perform great
    - Train on 10% of OpenImages
    - Do reconstruction (instead of denoising) ie: learn identity function
    - Use Ballé models as teacher (192 channels) and create a student from this architecture
    - Use weight and biases to log results
    - Future: experiment with different architectures and loss functions
- Apply KD to AE:
  - Find N, M values for "bmshj2018-hyperprior": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    - Learn how to use Weights and Biases
    - Rename conda env (balle_reproduction -> prim_env)
    - Access OpenImages dataset with TensorFlow Datasets
      - it does not work, storage limitation...

## 19/12/2024
- Meeting:
  - possibility to use other image dataset
  - Student number of channels 160/128
  - Improve training script (log other losses, fix checkpoint save...)
  - Train teacher and student alone
  - Train student with KD try different losses, architecture and add loss on latent hyperprior

# 11/01/2025
- Update README
- Train ScaleHyperPrior models (image reconstruction, no KD)
  - Teacher (N=128) / Student (N=64)
- Train TeacherAE and StudentAE models (image reconstruction, no KD)
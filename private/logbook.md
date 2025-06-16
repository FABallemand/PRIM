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
  - Read Microdosing: Knowledge Distillation for GAN-based Compression

## 13/12/2024
- Apply KD to AE:
  - Test teacher AE
  - Bad results
  - Noise scaled by 0.2 too high -> change for 0.1??
  - Train student AE
  - Train student AE with KD
- Understand KD:
  - Read and create reading notes
  - Read Microdosing: Knowledge Distillation for GAN-based Compression

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

## 11/01/2025
- Update README
- Train ScaleHyperPrior models (image reconstruction, teacher and no KD)
  - Teacher (N=128) / Student (N=64)
  - 254451
  - 254452
- Train TeacherAE and StudentAE models (image reconstruction, teacher and no KD)
- Understand LIC:
  - Read and create reading notes
  - Read End-to-end Optimized Image Compression

## 12/01/2025
- Train ScaleHyperPrior models (image reconstruction, student KD with and without latent loss KL)
- Train StudentAE models (image reconstruction, student KD)

## 13/01/2025
- Train ScaleHyperPrior models (image reconstruction, student KD with and without latent loss KL)
  - 256893
  - 256894
- Test TeacherAE and StudentAE models (image reconstruction)
  - Not impressive...

## 14/01/2025
- Understand LIC:
  - Read and create reading notes
  - Read End-to-end optimization of nonlinear transform codes for perceptual quality
- Update report 2
- Train ScaleHyperPrior models (image reconstruction, student KD with and without latent loss MSE)
  - 257451 (no latent loss)
  - 257452 (latent loss)

## 15/01/2025
- Meeting:
  - Need working code
  - Evaluate with PSNR during training
- Train multiple students with different architectures

## 13/01/2025
- Train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with latent loss MSE)
  - 258258
  - 258259
  - 258262
  - 258263
- Test ScaleHyperPrior models (image reconstruction, student KD with and without latent loss MSE)
  - 257451 (no latent loss)
  - 257451 (latent loss)

## 19/01/2025
- Test ScaleHyperPrior models (image compression, pre-trained teacher, student KD with latent loss MSE)
  - 258258
  - 258259
  - 258262
  - 258263
- fvcore to compute FLOPS
- Understand LIC:
  - Read and create reading notes
  - Read Variational Image Compression with a Scale Hyperprior

## 20/01/2025
- Understand LIC:
  - Read and create reading notes
  - Read Joint Autoregressive and Hierarchical Priors for Learned Image Compression
- Report:
  - Update intro
  - Update SOTA
- Train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with latent loss MSE)
  - 259782 (16)
  - 259783 (32)
  - 259784 (64)
  - 259785 (96)
  - 259786 (112)
- Train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with latent loss KL)
  - 259787 (16)

## 21/01/2025
- Train ScaleHyperprior for balle_reproduction
  - 260502
- Train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with latent loss KL)
  - 260532 (112)

  ## 22/01/2025
- Cancel train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with latent loss KL)
  - 259787 (16)
  - 260532 (112)
- Should have read doc: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
- Train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with latent loss KL)
  - 261095 (112)
- Meeting:
  - Need to use RateDistortionLoss
  - Compute model energy consumption
  - Do more experiments and send the code end of january for deeper training on super computer
- Model visualisation (torchview, torchviz, NN SVG, PlotNeuralNet)
- Compute energy consumption (https://pytorch.org/blog/zeus/)
- Compute latency
- Report:
  - Begin part 2

## 25/01/2025
- Test ScaleHyperPrior models (image compression, pre-trained teacher, student KD with latent loss MSE)
  - 259782 (16)
  - 259783 (32)
  - 259784 (64)
  - 259785 (96)
  - 259786 (112)
- Train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with RD loss and latent loss MSE)
  - 263674 (16)
  - 263687 (32)
  - 263688 (64)
  - 263690 (96)
  - 263691 (112)

## 26/01/2025
- Test ScaleHyperprior for balle_reproduction
  - 260502
- Report:
  - Update part 1

## 03/02/2025
- Test ScaleHyperPrior models (image compression, pre-trained teacher, student KD with RD loss and latent loss MSE)
  - 263674 (16)
  - 263687 (32)(112??)
  - 263688 (64)(112??)
  - 263690 (96)(112??)
  - 263691 (112)
- Train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with RD loss and latent loss MSE)
  - 274457 (32)
  - 274461 (64)
  - 274464 (96)
- Train TeacherAE and StudentAE models (image reconstruction, teacher and no KD)
  - 274518 (Teacher)
  - 274520 (Student no KD)
- Report:
  - Update part 2

## 04/02/2025
- Report:
  - Update part 2
- Estimate memory footprint (https://discuss.pytorch.org/t/finding-model-size/130275/2)

## 07/02/2025
- Train ScaleHyperPrior models (image compression, pre-trained teacher, student KD with RD loss and latent loss MSE, 64 channels, different lambda)
  - 280392 (0.0018)
  - 281662 (0.0035)
  - 281976 (0.0067)
  - 281979 (0.013)
  - x (0.025) (reuse 274461)

## 10/02/2025
- Test ScaleHyperPrior models (image compression, pre-trained teacher, student KD with RD loss and latent loss MSE)
  - 263674 (16)
  - 274457 (32)
  - 274461 (64)
  - 274464 (96)
  - 263691 (112)

## 12/02/2025
- Meeting:
  - Use energy per frame
  - Energy for decoder only
  - Use pynvml for energy consumption
  - Use GigaFLOPs instead of FLOPs
  - Look how energy consumption is computed with Zeus (sparse matrix multiplication, set all weights to zero)
  - For the report, explain why 64 channels is a good choice (reasonable, good tradeoff)

## 13/02/2025
- Improve testing:  
  - Use energy per frame

## 18/02/2025
- Improve testing:
  - Fix bugs
- Test ScaleHyperPrior models (image compression, pre-trained teacher, student KD with RD loss and latent loss MSE, 64 channels, different lambda)
  - 280392 (0.0018)
  - 281662 (0.0035)
  - 281976 (0.0067)
  - 281979 (0.013)
  - x (0.025) (reuse 274461)

## 19/02/2025
- Improve testing
- Meeting
- Report:
  - Update part 2

## 20/02/2025
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality 1, student KD with RD loss and latent loss MSE, 64 channels, different lambda)
  - 289751 (0.0018)
  - x (0.0035)
  - 289745 (0.0067)
  - x (0.013)
  - 289742 (0.025)
- Improve testing
- Report:
  - Update part 2

## 26/02/2025
- Test ScaleHyperPrior models (image compression, pre-trained teacher quality 1, student KD with RD loss and latent loss MSE, 64 channels, different lambda)
  - 289751 (0.0018)
  - x (0.0035)
  - 289745 (0.0067)
  - x (0.013)
  - 289742 (0.025)
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality 1, student KD with RD loss and latent loss MSE, 64 channels, different lambda)
  - 289751 (0.0018)
  - 295889 (0.0035)
  - 289745 (0.0067)
  - 296544 (0.013)
  - 289742 (0.025)
- Report:
  - Update part 2
  - Update figures

## 27/02/2025
- Report:
  - Update part 1

## 28/02/2025
- Report:
  - Update part 1
  - Update part 2
  - Update figures
- Train TeacherAE and StudentAE models (image reconstruction, student KD)
  - 298339

## 01/03/2025
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss KLD)
  - 299061 (16)
  - 299062 (64)
  - 299063 (112)
- Report

## 02/03/2025
- Report
- Pending tests

## 03/03/2025
- Pending tests
- Reports

## 04/03/2025
- Pending tests
- Reports
- Compare to codecs:
  - Test with PIL

## 07/03/2025
- Reports

## 08/03/2025
- Reports

## 08/03/2025
- Compare to codecs
- Reports

## 09/03/2025
- Reports

## 10/03/2025
- Reports

## 11/03/2025
- Reports

## 12/03/2025
- Reports
- Final presentation

## 11/04/2025
- Generate better barplots (correct colors)
- Try to fix cluster

## 14/04/2025
- GPU cluster (error with 3090 partition only...)
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss MSE, lmbda=(0.3, 0.3, 0.4))
  - 319228 (16)
  - 319229 (32)
  - 319230 (64)
  - 319231 (96)
  - 319233 (112)

## 11/05/2025
- Test ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss MSE, lmbda=(0.3, 0.3, 0.4))
  - 319228 (16)
  - 319229 (32)
  - 319230 (64)
  - 319231 (96)
  - 319233 (112)
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss MSE, 64 channels)
  - 319230 lmbda=(0.3, 0.3, 0.4)
  - 332617 lmbda=(0.4, 0.4, 0.2)
  - 332618 lmbda=(0.1, 0.1, 0.8)
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss MSE + hyper-latent loss MSE, lmbda=(0.2, 0.2, 0.2, 0.4))
  - ? (16)
  - ? (32)
  - ? (64)
  - ? (96)
  - ? (112)
- Need to fix issue (entropy bottleneck size depends on N...)

## 12/05/2025
- Fix issue with entropy bottle neck size (define N_entropy for student, to be considered for future comparisons)
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss MSE + hyper-latent loss MSE, lmbda=(0.2, 0.2, 0.2, 0.4))
  - ? (16)
  - ? (32)
  - ? (64)
  - ? (96)
  - ? (112)

## 13/05/2025
- Fix code
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss MSE + hyper-latent loss MSE, lmbda=(0.2, 0.2, 0.2, 0.4, 0.025))
  - ? (16)
  - ? (32)
  - 335897 (64)
  - ? (96)
  - ? (112)

## 15/05/2025
- Test ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss MSE, 64 channels)
  - 319230 lmbda=(0.3, 0.3, 0.4)
  - 332617 lmbda=(0.4, 0.4, 0.2)
  - 332618 lmbda=(0.1, 0.1, 0.8)
- Interesting results but requires zoom

## 16/05/2025
- Create script for hybrid KD
- Train ScaleHyperPrior models (image compression, pre-trained teacher quality r,d=1,5, student KD with RD loss, latent and hyper latent loss MSE, 64 channels)
  - 336202 lmbda=(0.2, 0.2, 0.2, 0.4, 0.025)

## 26/05/2025
- Test ScaleHyperPrior models (image compression, pre-trained teacher quality 5, student KD with RD loss and latent loss MSE + hyper-latent loss MSE, lmbda=(0.2, 0.2, 0.2, 0.4, 0.025))
  - ? (16)
  - ? (32)
  - 335897 (64)
  - ? (96)
  - ? (112)
- Test ScaleHyperPrior models (image compression, pre-trained teacher quality r,d=1,5, student KD with RD loss, latent and hyper latent loss MSE, 64 channels)
  - 336202 lmbda=(0.2, 0.2, 0.2, 0.4, 0.025)

## 29/05/2025
- Create figures of the latest experiments for paper

## 30/05/2025
- Update paper draft

## 31/05/2025
- Update paper draft

## TODO

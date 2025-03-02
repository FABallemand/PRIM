# 👨🏻‍🔬 Projet de Recherche et d’Innovation Master (PRIM)

## 🖼️ Learned Image Compression on FPGA
With the recent progress of machine learning, new compression algorithms appeared. These probabilistic algorithms are based on neural networks that can learn a better way to encode data with a minimum number of bits. The only issue being that these methods require way more processing power than previous state of the art compression algorithms. The goal of this project is to achieve real-time image compression on resource constrained platforms using frugal machine learning techniques such as pruning, quantisation and knowledge distillation.

## 🗂️ Reporistory Organisation
```
PRIM/
    ├── balle_bdpsnr/       # Reproducing SOTA results
    ├── balle_reproduction/ # Reproducing SOTA results
    ├── data/               # Datasets
    ├── kd_ae/              # Experiments with KD for image reconstruction
    ├── kd_ae_test/         # Experiments with KD for image reconstruction
    ├── kd_lic_experiments/ # Experiments with KD for LIC
    ├── private/            # Personal data
    ├── report_1/           # Intermediate report
    ├── report_2/           # Final report
    ├── README.md           # README
    └── requirements.txt    # Requirements
```

- The `balle_reproduction` contains the first step of the project, reproducing Ballé state-of-the-art results on a single model.
- The second step was to reproduce state-of-the-art results for different bit rate/quality tradeoffs, this is contained in the `balle_bdpsnr` folder.
- I then learned how to use knowledge distillation and tried to apply it on a simple autoencoder model for image denoising / reconstruction, the experiments and results can be found in `kd_ae_test`. Results are not impressive to say the least...
- Next, I proceeded in implementing knowledge distillation on state-of-the-art image reconstruction model but using them as auto-encoders. Training teacher and student model from scratch produced great visual results. The code and results can be found in `kd_ae`.
- Finally, I adapted the previous code to perform LIC. I experimented with different student architectures (number of channels). Code and results are in the `kd_lic_experiments` folder.

## 🔗 Related Links

## Learned Image Compression
- [Neural/Learned Image Compression: An Overview](https://medium.com/@loijilai_me/learned-image-compression-an-overview-625f3ab709f2)
- [End-to-end optimization of nonlinear transform codes for perceptual quality](https://arxiv.org/abs/1607.05006)
- [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704)
- [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436)
- [Joint Autoregressive and Hierarchical Priors for Learned Image Compression](https://arxiv.org/abs/1809.02736)
- [The Devil Is in the Details: Window-based Attention for Image Compression](https://arxiv.org/abs/2203.08450)
- [A Survey on Visual Transformer](https://arxiv.org/abs/2012.12556)

## Knowledge Distillation
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Microdosing: Knowledge Distillation for GAN based Compression](https://arxiv.org/abs/2201.02624)
- [Fast and High-Performance Learned Image Compression With Improved Checkerboard Context Model, Deformable Residual Module, and Knowledge Distillation](https://arxiv.org/abs/2309.02529)
- [Cross-Architecture Knowledge Distillation](https://arxiv.org/abs/2207.05273)
- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)

### Metrics
- [Structural similarity index measure](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)
- [Peak signal-to-noise ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [Bjøntegaard Delta (BD): A Tutorial Overview of the Metric, Evolution, Challenges, and Recommendations](https://arxiv.org/abs/2401.04039)

### Tools
- [CompressAI](https://interdigitalinc.github.io/CompressAI/zoo.html)
- [STF](https://github.com/Googolxx/STF) (GitHub reposirotry of "The Devil Is in the Details")
- [Bjøntegaard Delta (BD): A Tutorial Overview of the Metric, Evolution, Challenges, and Recommendations](https://arxiv.org/abs/2401.04039)
- [Bjontegaard_metric](https://github.com/Anserw/Bjontegaard_metric/tree/master)
- [Weights and Biases](https://wandb.ai/site/)
- [fvcore](https://github.com/facebookresearch/fvcore/tree/main)
- [zeus](https://ml.energy/zeus/)
- [pynvml](https://pypi.org/project/pynvml/)
- [Measuring GPU Energy: Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/#__tabbed_1_1)
- [Télécom Paris GPU Cluster Doc](https://docs.google.com/document/d/1lXykfpEUJCrbNh22D2f2kxNS0gV6t-j9A_juWFdiEnI/edit?tab=t.0)

## 👥 Author
- Fabien ALLEMAND
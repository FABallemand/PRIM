# Reading Notes

## Learned Image Compression

## Knowledge Distillation

### [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Ensemble learning improves results but is cumbursome
- Goal: knowledge of the ensemble in a single network
- Analogy with insects different forms for different use (larval form vs adult form)
- Same thing with ML: why use the same form to train and use, it is not the same goal and constrains!!
- Knowledge of the model are not the learnt weights, it is the mapping from a vector to another!!
- This "continuous" mapping contains more knowledge than simple binary target vectors
    - Car closer to truck than to bird, higher probability to mistake...
    - The continous mapping tells a lot about how the model tends to generalize
- Training on data that is not exactly the goal of the user is not an issue
    - We want models to generalize well but we train them on training data (not new data)
    - Large model that is the average of ensemble of models, small models trained on its mapping will generalise as good as the ensemble
- Idea: use output of teacher model as soft targets
    - Use training data or different "transfer" data
    - Student model can be trained on less data as soft targets contain (more entropy) more information and less variance in the gradient between training cases
- Issue with simple data (like MNIST)
    - Mapping too close to binary mapping (target)
    - Previous solution: do not use output of softmax but input of softmax for KD
    - Proposed solution: called distillation, raise the temperature of the final softmax until the teacher model produces a suitably soft set of targets
- Note: the transfer set can be unlabelled data!!
- How to use temperature in LIC??

Originally created in ... to achieve the same results than ensemble of models with a lower computational cost, knowledge distillation consists in transferring the knowledge of a cumbersome model into a single smaller one. In this approach the knowledge of a neural network is not represented by its weights but by the vector to vector mapping it has learned. A large "teacher" model can be trained with unlimited computing power for a long time on large datasets, then a smaller "student" model can learn the mapping of the teacher by using teh teacher's predictions as soft targets. This is different than training a smaller model alone, as the student model has a higher ability to generalize while requiring fewer and possibly unlabelled data. To compensate the lack of entropy (information) of the soft targets of simple tasks, the authors propose to use a temperature parameter to soften the teacher model output distribution.

### [Microdosing: Knowledge Distillation for GAN based Compression](https://arxiv.org/pdf/2201.02624)
- Context of image and video compression (internet traffic, pandamy)
- "At a high level, most of the methods can be understood as a sort of generative model that tries to
reconstruct the input instance from a quantized latent representation, coupled with a prior that is used
to compress these latents losslessl"
- Focus on low bitrate
- GAN-based framework -> good results but too costly (to train and use) especially the decoder part
- Good perceptual image quality with reduced size decoder using KD
- Very good for broadcasting (encoding once and decoding many times) possibly on edge devices
- Training student decoder from teacher decoder
    - Novel strategy for KD in the context of image and video compression
    - Based on sota HiFiC architecture
    - Proposal: replace the big general purpose texture generator with a smaller specific one (based on degradation aware blocks)
    - Only part of the decoder is changed and specific to the subset
    - The small specific decoder part is trained using KD on the encoder side before being sent
    - (Part specific to video)
- Overfitting student decoder to specific (set of) images
    - Instead of having a general purpose encoder/decoder, the input is split into subsets and the encoder send a specific decoder for every subset
    - ...
- Sending the specialized decoder weights alongside the image latents
    - ...

In the context of LIC, it is often assumed that the encoding task is performed on a single sender with unlimited resources. The latent representation is then broadcasted and decoded on many receivers with various constrains such as time (latency) and computing resources. Leveraging KD, a smaller and more efficient decoder can be trained while maintaining visual fidelity.

Noting that GAN-based LIC frameworks (like state of the art HiFiC) are able to reproduce texture using large general purpose networks, the approach proposed in ... overfits a smaller decoder network for every sequence of images that can be sent alongside the latents (more precisely only the blocs responsible of the texture decompression are replaced by a smaller bloc). The smaller decoder is trained using KD on the encoder side. This approach dramatically reduces the decoder model size and the decoding time while providing a great image quality.

### [Fast and High-Performance Learned Image Compression With Improved Checkerboard Context Model, Deformable Residual Module, and Knowledge Distillation](https://arxiv.org/pdf/2309.02529)
- "The main components of classical image compression standards, e.g., JPEG [1], JPEG 2000 [2], BPG (intra-coding of H.265/HEVC) [3], and H.266/VVC [4], include linear transform, quantization, and entropy coding. In the end-to-end learning-based framework, these components have been re-designed carefully."
- Detail current techniques of LIC
- Deformable Residual Module (DRM)
    - Deformable conv + residual bloc
    - Expand receptive field to obtain global information, capture and reduce spatial correlation of latent representations
    - Deformable convolution: conv with deformable receptive field
    - It helps extracting better features and representing objects
- Improved checkerboard context model
    - Divides latents into two subsets using a checkerboard pattern
    - Good for parallelism (subsets processed in parallel)
- Three-step KD
    - Tradeoff between performance and complexity of the decoder
    - Start from same architetures and then apply KD techniques to student to reduce its complexity
    - Step 1: train teacher network (encoder and decoder) with traditional loss function
    - Step 2: starting with a student network with the same architecture, train both network jointly with special loss function
    - Step 3: perform ablation on less relevant blocs of the decoder and train jointly with teacher once again
    - TRaining parameters: some lambdas decrease over time...
- L1 regularization
    - Make values of the latent representation sparser
- Related works (interesting)
- Big and complex architecture!!
- TODO: better understand the architecture...

The authors of ... propose four techniques to improve LIC with resource cautious decoders. They first imporve standard LIC by using deformable conviolution (convolution with a deformable receptive field) that helps extracting better features and representing objects. Then, a checkerboard context model is used to increase parallelism execution and a three-step KD method is used to reduce the decoder complexity (train teacher, train student with same architecture of the teacher jointly with the teacher, perform ablation on less relevant blocs of the student decoder and re-train jointly with teacher). Finally, L1 regularisation is introduced to make the latent representation sparser allowing to speed up decoding by only encoding non-zero channels in the encoding and decoding process, which can greatly reduce the encoding and decoding time at the cost of coding performance. The experimental results presented by the authors show better performance than traditional codes and state of the art LIC methods in both image quality and encoding-decoding time. 
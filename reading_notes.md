# Reading Notes

## Learned Image Compression

### [End-to-end optimization of nonlinear transform codes for perceptual quality](https://arxiv.org/abs/1607.05006)
- General framework for end-to-end optimization of the rate–distortion performance of nonlinear transform codes assuming scalar quantization
- Transform coding (all compression standards: invertible transformation, quantization, inverting the transformation)
- Comparison with the field of object and pattern recognition (used to be built manually but better performance thanks to end-to-end system optimisation using differentiable transformations and modern optimisation tools to jointly optimize over large datasets)
- Idea: do the same thing with image compression!!
- Framework of nonlinear transform coding which generalises the traditional paradigm
- Explaination of the framework [TO USE IN EXPLAINATION]
- Code rate measured with entropy
- Distortion traditionally measured in image space using MSE or PSNR (but PSNR does not align with human perception)
- Distortion in measured after applying a suitable "perceptual transform" (to achieve better approximation of perceptual visual distortion)
- Introduction of the loss function and lambda (tradeoff)
- Trick: add noise instead of quantization to ensure differentiability during training

Transform coding is a signal processing method that consists fo three steps: applying an invertible transformation to a signal, quantizing the transformed data to achieve a compact representation, and inverting the transform to recover an approximation of the original signal. This method is used by most deterministic image compression algorithms like JPEG and JPEG-2000.

In 2016, ... propose the first end-to-end optimised image compression framework. Inspired by the filed of object and pattern recognition, the framework leverages end-to-end optimisation to achieve better results than previous systems that were built by manually combining a sequence of individually designed and optimized processing stages. Still based on transform coding, the framework consists in transforming an image vector from the signal domain to a code domain vector using a differentiable nonlinear transformation (analysis transform) and applying scalar quantisation to achieve the compressed image. The code domain vector can be transformed back to the signal domain thanks to another differentiable nonlinear transformation (synthesis transform). Contrary to traditional methods, the synthesis transform is not necessariliy the exact inverse of the analysis transform as the system is jointly optimized over large datasets with the goal of minimising the rate-distortion loss. The rate is measured by the entropy of the quantized vector and the distortion, usually measured using MSE or PSNR in the signal space is evaluated with either MSE or NLP (normalized Laplacian pyramid) after applying a suitable perceptual transform to achieve better approximation of perceptual visual distortion. The authors propose transformations based on generalized divisive normlisation (and its approximate inverse) and to use additive uniform noise at training time to preserve the system differentiability. The first experiments conducted with this framework show substancial improvements in bit-rate and perceptual appearance compared to previous linear transform coding techniques.

### [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704)
- Image compression method with nonlinear analysis transformation, uniform quantizer, nonlinear synthesis transformation (conv and + nonlinear activation functions)
- Jointly optimize entire model for rate-distortion performance
- Better performance than JPEG and JPEG-2000 + improved visual quality
- Image compression -> link with entropy and probability
- Compression code -> finite entropy -> quantization -> lossy compression -> rate-distortion problem (depends on the use (storage, transmission))
- Transfrom coding (linear transformation, quantization, lossless entropy coding) like JPEG and JPEG-2000
- Framework for end-to-end optimization of an image compression model based
on nonlinear transforms (trained using MSE and GDN achitecture inspired by biological visual system) (architecture figure)
- Trick: quantization (zero gradient) is replaced by adding uniform noise during training
- Close to VAE
- Use of bitrate
- MSE used to comparison to other related works and because there was no reliable perceptual metric for color images
- Relation with variational generative models (authors started from rate-distortion optimization problem but their framework can be cast to variational problems)
- Yields unperfect but impressive (perceptual and measured (PSNR and MS-SSIM)) results (lack of detail but no artifcats like JPEG and JPEG-2000 on all images at all bit-rates)
- When bit-rate decreases quality progressively decrease (not the cas with JPEG and JPEG-2000, basis function)
- Possible improvement if trained with perceptual metric (not MSE)

In 2016, ... update the first end-to-end optimised image compression framework. Based on the same three-step transform coding method (linear transformation, quantization, lossless entropy coding) as deterministic image compression algorithms like JPEG and JPEG-2000, the proposed model uses a nonlinear analysis transformation, a uniform quantizer and a lossless entropy coding. It should be noted that the analysis transformation is inspired by biological visual systems and made of convolutions and nonlinear activation functions. By replacing quanitization by additive uniform noise at training time (where quantization would have cancelled gradients), the model is jointly optimised for rate-distortion performance using MSE. Although optimizing the model for a measure of perceptual distortion, would have exhibited visually superior performance, MSE was used in order to facilitate comparison with related works (usually trained with MSE) and because there was no reliable perceptual metric for color images. This novel framework yields unperfect but impressive results: details are lost in compression but it does not suffer from artifacts like JPEG and JPEG-2000. It outperforms JPEG and JPEG-2000 at all bit-rates both perceptually and quantitatively according to PSNR and MS-SSIM measures thanks to its ability of progressively reducing the image quality. [MENTION RATE INSTEAD OF ENTROPY IN ORIGINAL FRAMEWORK]

... update the framework. Based on the same three-step transform coding method (linear transformation, quantization, lossless entropy coding) as deterministic image compression algorithms like JPEG and JPEG-2000, the proposed model uses a nonlinear analysis transformation, a uniform quantizer and a lossless entropy coding. It should be noted that the analysis transformation is inspired by biological visual systems and made of convolutions and nonlinear activation functions. By replacing quanitization by additive uniform noise at training time (where quantization would have cancelled gradients), the model is jointly optimised for rate-distortion performance using bit-rate (instead of entropy)(more appropirate in the context of image compression) and MSE. Although optimizing the model for a measure of perceptual distortion, would have exhibited visually superior performance, MSE was used in order to facilitate comparison with related works (usually trained with MSE) and because there was no reliable perceptual metric for color images. This novel framework yields unperfect but impressive results: details are lost in compression but it does not suffer from artifacts like JPEG and JPEG-2000. It outperforms JPEG and JPEG-2000 at all bit-rates both perceptually and quantitatively according to PSNR and MS-SSIM measures thanks to its ability of progressively reducing the image quality.

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

### [Microdosing: Knowledge Distillation for GAN based Compression](https://arxiv.org/abs/2201.02624)
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

### [Fast and High-Performance Learned Image Compression With Improved Checkerboard Context Model, Deformable Residual Module, and Knowledge Distillation](https://arxiv.org/abs/2309.02529)
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

The authors of ... propose four techniques to improve LIC with resource cautious decoders. They first improve standard LIC by using deformable convolution (convolution with a deformable receptive field) that helps extracting better features and representing objects. Then, a checkerboard context model is used to increase parallelism execution and a three-step KD method is used to reduce the decoder complexity (train teacher, train student with same architecture of the teacher jointly with the teacher, perform ablation on less relevant blocs of the student decoder and re-train jointly with teacher). Finally, L1 regularisation is introduced to make the latent representation sparser allowing to speed up decoding by only encoding non-zero channels in the encoding and decoding process, which can greatly reduce the encoding and decoding time at the cost of coding performance. The experimental results presented by the authors show better performance than traditional codes and state of the art LIC methods in both image quality and encoding-decoding time. 
# Reading Notes

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
- Good perceptual image quality with reduced size decoder using KD
- 
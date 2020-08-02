# Segmentations-Leak: Membership Inference Attacks and Defenses in Semantic Image Segmentation

This folder contains the data splits used in our experiments.

The data splits are partitioned into two parts: dependent and independent.

For the dependent setting, we conduct experiments with the training and validation set from Cityscapes dataset.

For the independent setting, we train the shadow models with BDD100K and Mapillary datasets with our own splits, which are randomly selected.

Once the attackers are trained with shadow models, they are tested on the Cityscapes official split with 2975 training images and 500 validation images.

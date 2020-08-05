##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Segmentations-Leak: Membership Inference Attacks and Defenses in Semantic Image Segmentation
## by Yang He, Shadi Rahimian, Bernt Schiele and Mario Fritz, ECCV2020.
## The attacker (binary classification) implementation is based on third-party modules.
##
## Created by: Yang He - CISPA Helmholtz Center for Information Security
## Email: yang.he@cispa.saarland
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from .resnet import get_resnet

def get_resnet_classification_model(arch, pretrained=False, **kwargs):
    return get_resnet(arch, pretrained, **kwargs)

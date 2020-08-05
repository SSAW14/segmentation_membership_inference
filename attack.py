import argparse
import os
import random
import shutil
import time
import sys
import cv2
from PIL import Image

import numpy as np
import scipy.misc

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from models import get_resnet_classification_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser items for the configuration
parser.add_argument('-a', '--arch', type=str, default='resnet50',
                    help='model architecture')
parser.add_argument('--input-channel', default=3, type=int,
                    help='number of input channel')
parser.add_argument('-resume','--resume', default='./downloads/bdd100k_loss.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint')
parser.add_argument('-gpu', '--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-dilation', '--dilation', action='store_true',
                    help='Use dilated convolutions in attackers')

# parser items for membership inference attacks
parser.add_argument('-argmax', '--argmax', action='store_true',
                    help='attack a model trained with Argmax defense')
parser.add_argument('-gauss', '--gauss', default=0, type=float,
                    help='attack a model with Gauss defense')
parser.add_argument('-dpsgd', '--dpsgd', action='store_true',
                    help='attack a model trained with differential privacy SGD')
parser.add_argument('-num-patch', '--num-patch', default=6, type=int,
                    help='attack a model with Gauss defense')
parser.add_argument('-input', '--input', type=str, default='loss',
                    help="data representation for attacks. choose 'loss' or 'concate'.")

def main():
    args = parser.parse_args()

    assert not (args.gauss > 0 and args.dpsgd)
    assert not (args.gauss > 0 and args.argmax)
    assert not (args.argmax and args.dpsgd)

    # For Cityscapes label space with 19 classes. Concatenation leads to 38 input channels.
    args.input_channel = 1 if args.input == 'loss' else 38
    print("Use GPU: {}".format(args.gpu))
    print("=> using a model '{}' with {} input channel".format(args.arch, args.input_channel))

    # build a binary classifier as our per-patch attacher
    attacker = get_resnet_classification_model(arch=args.arch, input_channel=args.input_channel, num_classes=2, dilated=args.dilation)

    torch.cuda.set_device(args.gpu)
    attacker.cuda(args.gpu)

    checkpoint = torch.load(args.resume)
    attacker.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(args.resume))

    pred_dir = 'pred_dpsgd' if args.dpsgd else 'pred_sgd'
    tp, pos, score1 = test(attacker, './examples', 'train', pred_dir, args)
    tn, neg, score2 = test(attacker, './examples', 'val', pred_dir, args)
    score = np.concatenate((score1[:,0,1], score2[:,0,1]), axis=0)
    target = np.array([1] * pos + [0] * neg)
    print("AUC-score: {}".format(roc_auc_score(target, score)))


def SLM(pred, label, ignore=255):
    sz = label.shape
    loss = np.zeros((1, sz[0] , sz[1]))
    for i in np.unique(label):
        if i == ignore:
            continue

        mask = (label == i)

        tmp = pred[i, :, :] * mask
        tmp[tmp < 1e-30] = 1e-30
        tmp = -np.log(tmp)

        loss[0, mask] = tmp[mask]
    return loss

def Argmax(pred):
        post_tmp = np.zeros(pred.shape)
        order = np.argsort(pred, axis=0)
        order = order[::-1,:,:]

        for i in range(order.shape[0]):
            mask = order[0,:,:] == i
            post_tmp[i,mask] = 1

        pred = post_tmp
        return pred

def Label2Tensor(label, num_class=19):
        dims = label.shape

        output = np.zeros((num_class,dims[0],dims[1]), dtype='float32')

        for i in range(0, num_class):
            mask = label == i
            output[i,mask] = 1

        return output

def test(model, data_dir, membership, pred_dir, args):
    # switch to eval mode
    model.eval()

    image_dir = os.path.join(data_dir, membership, 'img')
    label_dir = os.path.join(data_dir, membership, 'label')
    pred_dir = os.path.join(data_dir, membership, pred_dir)

    files = os.listdir(label_dir)
    files.sort()

    target = 1 if membership == 'train' else 0
    correct = 0
    total = len(files)

    score = []

    for file in files:
        name = file.split('.')[0]
        #image = np.array(Image.open('%s/%s.png'%(image_dir, name)))
        label = np.array(Image.open('%s/%s.png'%(label_dir, name)))
        segment = np.load('%s/%s_%s.npy'%(pred_dir, name, membership ))

        if args.gauss > 0:
            gaussian_noise = np.random.normal(0, args.gauss, segment.shape)
            segment = segment + gaussian_noise
            segment[segment < 1e-6] = 1e-6

        segment = segment / np.repeat(np.sum(segment, 0).reshape(1,segment.shape[1],segment.shape[2]), segment.shape[0], axis=0 )

        if args.argmax:
            segment = Argmax(segment)

        #import ipdb; ipdb.set_trace()
        label = label[::8,::8]
        input = []
        if args.input == 'loss':
            input = SLM(segment, label)
            if args.argmax:
                input = input / 50
        elif args.input == 'concate':
            label = Label2Tensor(label)
            input = np.concatenate((segment, label), axis=0)
        #import ipdb; ipdb.set_trace()
        input = torch.from_numpy(input).float().unsqueeze(0).cuda(args.gpu)
        pid = 0
        pred = 0
        while pid < args.num_patch:
            bias_x = np.random.randint(input.shape[2] - 90)
            bias_y = np.random.randint(input.shape[3] - 90)
            input_tmp = input[:,:,bias_x:bias_x+90,bias_y:bias_y+90]
            output, feat = model(input_tmp)
            pid += 1
            pred += output

        #import ipdb; ipdb.set_trace()
        score.append(pred.cpu().detach().numpy())
        pred = (pred[:,0] < pred[:,1])
        correct += int(pred == target)
        #if int(pred == target):
        #    print('correct')
        #else:
        #    print('incorrect')

    return correct, total, np.array(score)

if __name__ == '__main__':
    main()


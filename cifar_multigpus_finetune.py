
import argparse
import os
import sys

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torchvision.models import resnet152, regnet_x_32gf, wide_resnet101_2

from collections import OrderedDict
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import tqdm
import copy
import json

from train import *
from test import *


def main():

    print("Running command:", str(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        help="Input MLP-Mixer model architecture.",
        type=str,
        choices=["resnet152", "regnet_x_32gf", "wide_resnet101_2"],
        default='B_16',
    )
    parser.add_argument(
        "--dataset",
        help="Specify dataset",
        choices=["cifar10", "cifar100"],
        default="cifar10",
    )
    parser.add_argument(
        "--dataset_path",
        help="Specify the path of the dataset",
        default="dataset/",
    )
    parser.add_argument(
        "--cp",
        help="Input checkpoints path.",
        default=None,
    )
    parser.add_argument(
        "--E",
        type=int,
        help="Maxium number of epochs to train.",
        default=5,
    )
    parser.add_argument(
        "--LR",
        type=float,
        help="Learning rate for training input transformation of training clean model.",
        default=5,
    )
    parser.add_argument(
        "--BS",
        type=int,
        help="Training batch size.",
        default=128,
    )
    parser.add_argument(
        "--TBS",
        type=int,
        help="Test batch size.",
        default=100,
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        help="Loading pertrained model or not.",
        default=True,
    )
    parser.add_argument(
        "--finetune",
        type=bool,
        help="Finetune the model or not.",
        default=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Training mode or testing mode.",
        choices=["train", "test"],
        default='train',
    )

    args = parser.parse_args()

    print("Preparing data..", args.dataset)
    if args.dataset == "cifar10":
        classes = 10
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((224, 224)), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=args.dataset_path,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.BS, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.CIFAR10(
            root=args.dataset_path,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.TBS,
            shuffle=False,
            num_workers=2,
        )
    elif args.dataset == "cifar100":
        dataset = "cifar100"
        classes = 100
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root=args.dataset_path,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.BS, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.CIFAR100(
            root=args.dataset_path,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.TBS,
            shuffle=False,
            num_workers=2,
        )
    # Start finetuning the model with specific dataset and using quantization-aware training
    if args.mode == 'train':
        print('========== Start finetuning the model with specific dataset ==========')
        training(trainloader, args.arch, args.dataset, classes, args.LR, args.E, args.pretrained, args.finetune)
    if args.mode == 'test':
        # Start testing the model with specific dataset.
        print('========== Start testing the model with specific dataset. ==========')
        testing(trainloader, testloader, args.arch, args.cp, args.dataset, classes)

if __name__ == "__main__":
    main()
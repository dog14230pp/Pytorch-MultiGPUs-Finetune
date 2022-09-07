
import argparse
import os
import sys

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import tqdm
from torchvision.models import resnet152, regnet_x_32gf, wide_resnet101_2

def testing(trainloader, testloader, arch, cp, dataset, classes):
    """
      Calculating the accuracy with given clean model and perturbed model.
      :param testloader: The loader of testing data.
      :param transform_model: The object of transformation model.
      :param arch: The architecture of the MLP-Mixer.
      :param cp: The path of the checkpoints would be loaded.
      :param dataset: The specific dataset.
      :param classes: The classes of the dataset.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch == 'resnet152':
        model = resnet152()
    elif arch == 'regnet_x_32gf':
        model = regnet_x_32gf()
    elif arch == 'wide_resnet101_2':
        model = wide_resnet101_2()

    model.fc = nn.Linear(model.fc.in_features, classes)

    print(model)

    model_weights_dict_old = torch.load(cp)['model_state_dict']
    model_weights_dict = dict()
    for key, value in model_weights_dict_old.items():
        model_weights_dict[key.replace('module.', '')] = value

    model.load_state_dict(model_weights_dict)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    total_train = 0
    total_test = 0
    correct_orig_train = 0
    correct_p_train = 0
    correct_orig_test = 0
    correct_p_test = 0

    # For training data:
    for x, y in tqdm.tqdm(trainloader):
        total_train += 1
        x, y = x.to(device), y.to(device)
        out_orig = model(x)
        _, pred_orig = out_orig.max(1)
        y = y.view(y.size(0))
        correct_orig_train += torch.sum(pred_orig == y.data).item()
    accuracy_orig_train = correct_orig_train / (len(trainloader.dataset))

    # For testing data:
    for x, y in tqdm.tqdm(testloader):
        total_test += 1
        x, y = x.to(device), y.to(device)
        out_orig = model(x)
        _, pred_orig = out_orig.max(1)
        y = y.view(y.size(0))
        correct_orig_test += torch.sum(pred_orig == y.data).item()
    accuracy_orig_test = correct_orig_test / (len(testloader.dataset))

    print("Accuracy of training data: clean model: {:5f}".format(accuracy_orig_train))
    print("Accuracy of testing data: clean model: {:5f}".format(accuracy_orig_test))
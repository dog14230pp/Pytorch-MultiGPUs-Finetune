
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

def training(trainloader, arch, dataset, classes, learning_rate, epochs, pretrained, finetune):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained:
        if arch == 'resnet152':
            model = resnet152(weights="IMAGENET1K_V2")
        elif arch == 'regnet_x_32gf':
            model = regnet_x_32gf(weights="IMAGENET1K_V2")
        elif arch == 'wide_resnet101_2':
            model = wide_resnet101_2(weights="IMAGENET1K_V2")
        if finetune:
            model.fc = nn.Linear(model.fc.in_features, classes)
            print('++++++++++ Start Finetune Mode ++++++++++')
    else:
        if arch == 'resnet152':
            model = resnet152()
        elif arch == 'regnet_x_32gf':
            model = regnet_x_32gf()
        elif arch == 'wide_resnet101_2':
            model = wide_resnet101_2()

    model = nn.DataParallel(model)
    model = model.to(device)
    model.train()

    # summary(model, (3, 224, 224))    

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    for epoch in range(epochs):

        print("Epoch: %03d" % epoch)

        running_loss = 0.0
        running_correct = 0
        for batch_id, (inputs, outputs) in tqdm.tqdm(enumerate(trainloader)):

            inputs = inputs.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()

            model_outputs = model(inputs)  

            _, preds = torch.max(model_outputs, 1)
            outputs = outputs.view(
                outputs.size(0)
            )  # changing the size from (batch_size,1) to batch_size.

            loss = nn.CrossEntropyLoss()(model_outputs, outputs)

            # Compute gradient of perturbed weights with perturbed loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            running_correct += torch.sum(preds == outputs.data)

        accuracy = running_correct.double() / (len(trainloader.dataset))
        print("For epoch: {}, loss: {:.6f}, accuracy: {:.5f}".format(epoch, running_loss / len(trainloader.dataset), accuracy))

        if (epoch+1)%20 == 0 or (epoch+1) == epochs:

            extra = [arch, dataset, str(epoch+1)]

            model_path = os.path.join("model_weights/", arch, dataset, "_".join(extra) + ".pth")

            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))

            if os.path.exists(model_path):
                print("Checkpoint already present ('%s')" % model_path)
                sys.exit(1)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss / batch_id,
                    "accuracy": accuracy,
                },
                model_path,
            )
import pandas as pd
import numpy as np

import torch
from torch import nn, from_numpy, tensor, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
import json
import argparse
import time
import os
import project_utils


# takes input about architecture of model, hyperparameters, and GPU/CPU, learning_parameters

def get_args():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('--data_dir', default="flowers", help="path of train dataset")
    parser.add_argument('--arch', dest="arch", default="resnet50", type = str, help = "choose model to train the dataset")
    parser.add_argument('--gpu', dest="gpu", action="store_true", default="gpu", help = "Use GPU or CPU to train model")
    parser.add_argument('--checkpoint_final', dest="checkpoint_final", action="store", default="./checkpoint_final.pth", help = "saving the model")
    parser.add_argument('--lr', dest="lr", type=float, default=0.003, help = "learning rate")
    parser.add_argument('--dropout', dest = "dropout", type=float, default = 0.2, help = "set the dropout probability")
    parser.add_argument('--epochs', dest="epochs", type=int, default=4, help = "set number of epochs")
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", default=512)
    parser.add_argument('--cat_names', dest="cat_names", default='cat_to_name.json')

    return parser.parse_args()

in_arg = get_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_data, batch_size=32)
valid_dataloaders = torch.utils.data.DataLoader(valid_data, batch_size=32)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#count the number of layers
res = len(cat_to_name)


data_dir = in_arg.data_dir
arch = in_arg.arch
gpu = in_arg.gpu
checkpoint = in_arg.checkpoint_final
lr = in_arg.lr
dropout = in_arg.dropout
epochs = in_arg.epochs
hidden_units = in_arg.hidden_units


def Classifier(arch='resnet50', dropout=0.2):
    if arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_size = 2048

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(2048, 512), 
                               nn.ReLU(), 
                               nn.Dropout(p=0.2), 
                               nn.Linear(512, res), 
                               nn.LogSoftmax(dim=1)
                               )

    model.fc = classifier

    return model


#Establishs the model, criterion, and optimizer
model = Classifier(arch, dropout)


#Defines the function to train the defined model
def train_model(model=model, trainloader=train_dataloaders, validloader=valid_dataloaders, epochs=4,print_every=10):

    # training the model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 4
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    criterion = nn.NLLLoss() 
    print_every = 10

    running_loss = 0

    for epoch in range(epochs):
        steps = 0
    
        for inputs, labels in train_dataloaders:
            start = time.time()
        
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()    
        

            #with torch.no_grads():
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                for inputs, labels in valid_dataloaders:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
                end = time.time()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_dataloaders):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_dataloaders):.3f}.. "
                      f"Time taken: {end - start:.3f}"
                      )
            
                running_loss = 0
                model.train()    

train_model(model, train_dataloaders, valid_dataloaders, epochs)


def test_checker(testloader=test_dataloaders):
    test_loss = 0
    accuracy = 0
    model.eval()

    start = time.time()

    for inputs, labels in test_dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)        
    
        logps = model(inputs)
        loss = criterion(logps, labels)

        test_loss += loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() 
    
    end = time.time()
    
    print(f"Test loss: {test_loss/len(test_dataloaders):.3f}.. "
          f"Test accuracy: {accuracy/len(test_dataloaders):.3f}.. "
          f"Time taken: {end - start:.3f}"
          )    


#Runs accuracy checker functional
test_checker(test_dataloaders)



#Saving the checpoint to the save_dir path
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 2048,
              'output_size': 102,
              'arch': model,
              'fc' : classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint_final0.pth')
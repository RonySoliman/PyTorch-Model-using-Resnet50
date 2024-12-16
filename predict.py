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
import matplotlib.pyplot as plt


# The get_input_args function uses argparse to define the inputs 
def get_args():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--input_img', default="flowers/test/1/image_04938.jpg", help = "flowers", metavar="flowers")
    parser.add_argument('--gpu', action="store_true", default="gpu", help = "UseGPU_or_CPU_toTrainModel")
    parser.add_argument('--checkpoint_final', action="store", default="./checkpoint_final.pth", help = "checkpoint_final")
    parser.add_argument('--top_k', default=5, dest="top_k", type=int)
    parser.add_argument('--cat_names', dest="cat_names", default='cat_to_name.json')

    return parser.parse_args()

in_arg = get_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

input_img = in_arg.input_img
gpu = in_arg.gpu
checkpoint = in_arg.checkpoint_final
topk = in_arg.top_k
cat_names = in_arg.cat_names

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = checkpoint['arch']
    model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model

#
model = load_checkpoint('checkpoint_final.pth')
print(model)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img_pil = Image.open(image)

    # define transforms

    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])
                                    ])
    
    img_tensor = preprocess(img_pil)
    np_image = np.array(img_tensor)
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

import random

img = random.choice(os.listdir('./flowers/test/69/'))
img_path = './flowers/test/69/' + img

imshow(process_image(img_path))

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = from_numpy(process_image(image_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, image = model.to(device), image.to(device, dtype=torch.float)
    model.eval()
    
    output = model(image.unsqueeze(0)) 
    ps = torch.exp(output)
    
    # getting the topk (=5) probabilites and indexes
    prob = torch.topk(ps, topk)[0].tolist()[0] # probabilities
    index = torch.topk(ps, topk)[1].tolist()[0] # index
    
    idx = []
    for i in range(len(model.class_to_idx.items())):
        idx.append(list(model.class_to_idx.items())[i][0])
        
    classes = []
    for i in range(topk):
        classes.append(idx[index[i]])
    
    return prob, classes

img = random.choice(os.listdir('./flowers/test/23/'))
img_path = './flowers/test/23/' + img

imshow(process_image(img_path))


img = random.choice(os.listdir('./flowers/test/44/'))
img_path = './flowers/test/44/' + img
image = process_image(img_path)
prob, classes = predict(img_path, model)

plt.figure(figsize=(6,10))

ax1 = plt.subplot(2, 1, 1)
imshow(image, ax1)

labels = []
for cl in classes:
    labels.append(cat_to_name[cl])

ax2 = plt.subplot(2, 1, 2)
y_pos = np.arange(5)
ax2.set_yticks(y_pos, labels=labels)
ax2.set_xlabel('Probability')
ax2.invert_yaxis() 
ax2.barh(y_pos, prob, align='center')

plt.tight_layout()
plt.show()


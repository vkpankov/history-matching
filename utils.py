import re
from transformer_net import TransformerNet
import torch
import numpy as np
import random
import math
import array

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def load_model(file_name):
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(file_name, map_location=torch.device('cuda'))
                # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model = style_model.to("cuda")
        return style_model

def plotHardData(ax, val_sand=1, val_mud=0):
    hard_data = np.loadtxt("hard_as_hard_120.dat", skiprows=7)
    cdict = {val_sand: 'red', val_mud: 'blue'}
    hard_data_colors = [cdict[d] for d in hard_data[:,3]]
    ax.scatter(hard_data[:,0],hard_data[:,1],c=hard_data_colors)
    ax.xticks(hard_data[:,0])
    ax.yticks(hard_data[:,1])
    
def plotWells(ax, injectors, producers):
    i = 1
    j = 1
    for p in injectors:
        ax.plot(p[0],p[1],'^',color='blue')
        ax.text(x=p[0]+1, y=p[1]+1, s=f"{i}", fontdict=dict(color='black', alpha=1, size=12), bbox=dict(facecolor='white',alpha=0.5))
        i+=1
    for p in producers:
        ax.plot(p[0],p[1],'o',color='red')
        ax.text(x=p[0] + 1, y=p[1] + 1, s=f"{j}", fontdict=dict(color='black', alpha=1, size=12), bbox=dict(facecolor='white',alpha=0.5))
        j+=1


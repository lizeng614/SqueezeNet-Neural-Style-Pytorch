#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

# default arguments
OUTPUT_PATH =  'result.jpg'
ITERATIONS = 500
CONTENT_WEIGHT = 1e-3
STYLE_WEIGHT = 1
LEARNING_RATE = 1e-1
REPORT_INTERVAL = 10
IMSAVE_INTERVAL = 100
STANDARD_TRAIN = False
STYLE_RESIZE = 224
#parser
def build_parser():
    desc = 'a light pytorch implementation of neural style using squeezeNet.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--content',
            dest='content', help='path to content image', required=True)
    parser.add_argument('--style',
            dest='style', help='path to style image', required=True)
    parser.add_argument('--output', dest='output', 
            help='output path e.g. output.jpg', default = OUTPUT_PATH)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            default=STYLE_WEIGHT)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            default=CONTENT_WEIGHT)
    parser.add_argument('--content-resize', type = int, dest='content_resize',
            help='resize so that size of smaller edge match the given size', 
            default = None)
    parser.add_argument('--style-resize', type = int, dest='style_resize',
            help='resize so the size of smaller edge match the given size',
            default = STYLE_RESIZE)
    parser.add_argument('--iterations', type=int, dest='iters', 
            help='iterations (default %(default)s)', default=ITERATIONS)
    parser.add_argument('--learning-rate', type=float, dest='lr', 
            help='learning-rate for adam', default = LEARNING_RATE)
    parser.add_argument('--report-interval', type=int, dest='report_intvl', 
            help='report loss and current image every interval number', 
            default = REPORT_INTERVAL)
    parser.add_argument('--imsave-interval', type=int, dest='imsave_intvl', 
            help='save image every interval number', default = IMSAVE_INTERVAL)
    parser.add_argument('--standard-train',type=bool,dest='std_tr',
            help='standard train: lr=0.1 for 500 step and then lr=0.01 for 500 \
            step', default=STANDARD_TRAIN)
    return parser

### some functions to deal with image
def imload(image_name,resize = None):
    # a function to load image and transfer to Pytorch Variable.
    image = Image.open(image_name)
    if resize is not None:
        resizefunc = transforms.Scale(resize)
        image = resizefunc(image)
    transform = transforms.Compose([
        transforms.ToTensor(),#Converts (H x W x C) of[0, 255] to (C x H x W) of range [0.0, 1.0]. 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    image = Variable(transform(image),volatile=True)
    image = image.unsqueeze(0) 
    return image

def imshow(img):
    # convert torch tensor to PIL image and then show image inline.
    img=toImage(img[0].data*0.5+0.5) # denormalize tensor before convert
    plt.imshow(img,aspect=None)
    plt.axis('off')
    plt.gcf().set_size_inches(8, 8)
    plt.show()

def imsave(img,path):
    # convert torch tensor to PIL image and then save to path
    img=toImage(img[0].data*0.5+0.5) # denormalize tensor before convert
    img.save(path)    

### classes and functions to build the net 
class FeatureExtracter(nn.Module):
    # a nn.Module class to extract a intermediate activation of a Torch module
    def __init__(self,submodule):
        super().__init__()
        self.submodule = submodule
    def forward(self,image,layers):
        features = []
        for i in range(layers[-1]+1):
            image = self.submodule[i](image)
            if i in layers :
                features.append(image)       
        return features

class GramMatrix(nn.Module):
    # a nn.Module class to build gram matrix as style feature
    def forward(self,style_features):
        gram_features=[]
        for feature in style_features:
            n,f,h,w = feature.size()
            feature = feature.resize(n*f,h*w)
            gram_features.append((feature@feature.t()).div_(2*n*f*h*w))
        return gram_features

class Stylize(nn.Module): 
    def forward(self,x):
        s_feats = feature(x,STYLE_LAYER)
        s_feats = gram(s_feats)
        c_feats = feature(x,CONTENT_LAYER)
        return s_feats,c_feats

def totalloss(style_refs,content_refs,style_features,content_features,style_weight,content_weight):
    # compute total loss 
    style_loss = [l2loss(style_features[i],style_refs[i]) for i in range(len(style_features))] 
    #a small trick to balance the influnce of diffirent style layer
    mean_loss = sum(style_loss).data[0]/len(style_features)
    
    style_loss = sum([(mean_loss/l.data[0])*l*STYLE_LAYER_WEIGHTS[i] 
                    for i,l in enumerate(style_loss)])/len(style_features) 
    
    content_loss = sum([l2loss(content_features[i],content_refs[i]) 
                    for i in range(len(content_refs))])/len(content_refs)
    total_loss = style_weight*style_loss+content_weight*content_loss
    return total_loss

def reference(style_img,content_img):
    # a function to compute style and content refenrences as used for loss
    style_refs = feature(style_img,STYLE_LAYER)
    style_refs = gram(style_refs)
    style_refs = [Variable(i.data) for i in style_refs]
    content_refs = feature(content_img,CONTENT_LAYER)
    content_refs = [Variable(i.data) for i in content_refs]
    return style_refs,content_refs
###############################################################################
# set net parameter
STYLE_LAYER =[1,2,3,4,6,7,9]# could add more,maximal to 12
STYLE_LAYER_WEIGHTS = [21,21,1,1,1,7,7]# this should be same length as STYLE_LAYER
CONTENT_LAYER = [1,2,3]
options = build_parser().parse_args()

#load the image
content_img = imload(options.content,options.content_resize)
style_img = imload(options.style,options.style_resize)

#load load pretrained home work
model = models.squeezenet1_1(pretrained=True)
SUBMODEL = next(model.children())
feature = FeatureExtracter(SUBMODEL)
gram = GramMatrix()
#build net component and useful function
style = Stylize()
l2loss = nn.MSELoss(size_average=False)
toImage = transforms.ToPILImage()

#init a trainable image
train_img = Variable(torch.randn(content_img.size()),requires_grad=True)

#optimizer
optimizer = optim.Adam([train_img],lr = options.lr)
optimizer1 = optim.Adam([train_img],lr = 0.1)
optimizer2 = optim.Adam([train_img],lr = 0.01)

#trakers
loss_history = []
min_loss = float('inf')
best_img = None

#reference to compute loss
style_refs,content_refs = reference(style_img,content_img)

Start = datetime.now()
num_iters = 1000 if options.std_tr else options.iters

for i in range(num_iters):
    
    optimizer.zero_grad()
    
    train_img.data.clamp_(-1,1) # useful at first sevaral training step
    
    style_features,content_features = style(train_img)
    
    loss = totalloss(style_refs,content_refs,style_features,content_features,
                     options.style_weight,options.content_weight)
    
    loss_history.append(loss.data[0])
    
    # save best image result before image update
    if min_loss > loss_history[-1]:
        min_loss = loss_history[-1]
        best_img = train_img
    
    loss.backward()
    
    if options.std_tr :
        if i<=500 :
            optimizer1.step() 
        else :
            optimizer2.step()
    else:
        optimizer.step()
    
    # report loss and image  
    if i % options.report_intvl == 0:
        print("step: %d loss: %f,time per step:%s s" 
            %(i,loss_history[-1],(datetime.now()-Start)/options.report_intvl))
        Start = datetime.now()
    # save image every imsave_intvl
    if i % options.imsave_intvl == 0 and i !=0:
        save_path = options.output.replace('.jpg','_step%d.jpg'%i)
        imsave(train_img,save_path)
        print("image at step %d saved."%i)

#save the best image
imsave(best_img,options.output)

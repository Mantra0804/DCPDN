from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from misc import *
import pdb
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from misc import *
import dehaze22  as net


import pdb
import torchvision.models as models
import h5py
import torch.nn.functional as F
from skimage import measure
import numpy as np

parser = argparse.ArgumentParser()
"""parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')"""
dataset = 'pix2pix'

parser.add_argument('--dataroot', required=False,
  default='./nat_new4', help='path to trn dataset')
dataroot = './nat_new4'

#parser.add_argument('--valDataroot', required=False,
#  default='./nat_new4', help='path to val dataset')
valDataroot = './nat_new4'

#parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
mode = 'B2A'

#parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
batchSize = 1

#parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
valBatchSize = 1

#parser.add_argument('--originalSize', type=int,
#  default=1024, help='the height / width of the original input image')
originalSize = 1024

#parser.add_argument('--imageSize', type=int,
#  default=1024, help='the height / width of the cropped input image to network')
imageSize = 1024

inputChannelSize=3
outputChannelSize = 3
ngf=64
ndf = 64
"""
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
"""

niter=400
lrD=0.0002
lrG=0.0002
annealStart=0
annealEvery=400
lambdaGAN=0.01
lambdaIMG=1
poolSize=50
wd=0.0000
beta1=0.5
workers=0
exp='sample'
display=5
evalIter=500


#create_exp_dir(opt.exp)
manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
#random.seed(opt.manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
print("Random Seed: ", manualSeed)

# get dataloader
dataloader = getLoader(dataset,
                       dataroot,
                       originalSize,
                       imageSize,
                       batchSize,
                       workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=manualSeed)
dataset='pix2pix_val'

valDataloader = getLoader(dataset,
                          valDataroot,
                          imageSize, #opt.originalSize,
                          imageSize,
                          valBatchSize,
                          workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='Train',
                          shuffle=False,
                          seed=manualSeed)

# get logger
trainLogger = open('%s/train.log' % exp, 'w')

def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_y




netG = net.dehaze(inputChannelSize, outputChannelSize, ngf)
netG.load_state_dict(torch.load('netG.pth'))



#if opt.netG != '':
#  netG.load_state_dict(torch.load(opt.netG))
#print(netG)



netG.train()


target= torch.FloatTensor(batchSize, outputChannelSize, imageSize,imageSize)
input = torch.FloatTensor(batchSize, inputChannelSize, imageSize, imageSize)




val_target= torch.FloatTensor(valBatchSize, outputChannelSize,imageSize, imageSize)
val_input = torch.FloatTensor(valBatchSize, inputChannelSize, imageSize, imageSize)
label_d = torch.FloatTensor(batchSize)


target = torch.FloatTensor(batchSize, outputChannelSize,  imageSize,  imageSize)
input = torch.FloatTensor( batchSize, inputChannelSize,  imageSize,  imageSize)
depth = torch.FloatTensor( batchSize, inputChannelSize,  imageSize,  imageSize)
ato = torch.FloatTensor( batchSize, inputChannelSize,  imageSize,  imageSize)


val_target = torch.FloatTensor( valBatchSize, outputChannelSize,  imageSize,  imageSize)
val_input = torch.FloatTensor( valBatchSize, inputChannelSize,  imageSize,  imageSize)
val_depth = torch.FloatTensor( valBatchSize, inputChannelSize,  imageSize,  imageSize)
val_ato = torch.FloatTensor( valBatchSize, inputChannelSize,  imageSize,  imageSize)



# NOTE: size of 2D output maps in the discriminator
sizePatchGAN = 30
real_label = 1
fake_label = 0

# image pool storing previously generated samples from G
imagePool = ImagePool(poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)

netG.cuda()

target, input, depth, ato = target.cuda(), input.cuda(), depth.cuda(), ato.cuda()
val_target, val_input, val_depth, val_ato = val_target.cuda(), val_input.cuda(), val_depth.cuda(), val_ato.cuda()

target = Variable(target, volatile=True)
input = Variable(input,volatile=True)
depth = Variable(depth,volatile=True)
ato = Variable(ato,volatile=True)

label_d = Variable(label_d.cuda())

import numpy
import math

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    # PIXEL_MAX = 1

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
import time


# NOTE training loop
ganIterations = 0
index=0
psnrall = 0
ssimall=0
iteration = 0
# print(1)
for epoch in range(1):
  for i, data in enumerate(valDataloader, 0):
    t0 = time.time()

    if mode == 'B2A':
        input_cpu, target_cpu, depth_cpu, ato_cpu, imgname = data
    elif mode == 'A2B' :
        input_cpu, target_cpu, depth_cpu, ato_cpu, imgname = data
    batch_size = target_cpu.size(0)
    # print(i)
    target_cpu, input_cpu, depth_cpu, ato_cpu = target_cpu.float().cuda(), input_cpu.float().cuda(), depth_cpu.float().cuda(), ato_cpu.float().cuda()
    # get paired data
    target.resize_as_(target_cpu).copy_(target_cpu)
    input.resize_as_(input_cpu).copy_(input_cpu)
    depth.resize_as_(depth_cpu).copy_(depth_cpu)
    ato.resize_as_(ato_cpu).copy_(ato_cpu)
    #


    x_hat, tran_hat, atp_hat, dehaze2= netG(input)

    zz=x_hat.data

    iteration=iteration+1

    index2 = 0
    directory='./result_cvpr18/image/real_dehazed/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(valBatchSize):
        index=index+1
        print(index)
        zz1=zz[index2,:,:,:]

        vutils.save_image(zz1, './result_cvpr18/image/real_dehazed/'+imgname[index2]+'_DCPCN.png', normalize=True, scale_each=False)
trainLogger.close()

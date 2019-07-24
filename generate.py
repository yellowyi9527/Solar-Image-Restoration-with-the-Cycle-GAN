from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
from utils.dataset import DATASET 
from model.Discriminator import Discriminator
from model.Generator import Generator
from utils.fitsFun import DATASET_fits,DATASET_fits_sort2
from utils.fitsFun import LoadSaveFits

parser = argparse.ArgumentParser(description='train pix2pix model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='samples/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--bludataPath', default='/home/lab30201/yellow/data_project/cycleGANdeconvolution/data/T_disentangle/test/', help='' )
parser.add_argument('--oridataPath', default='/home/lab30201/yellow/data_project/cycleGANdeconvolution/data/T_disentangle/or/', help='' )
parser.add_argument('--loadSize', type=int, default=256, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=256, help='random crop image to this size')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=1, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=1, help='channel number of output image')
parser.add_argument('--G_AB', default='./checkpoints/G_AB_12000.pth', help='path to pre-trained G_AB')
parser.add_argument('--G_BA', default='./checkpoints/G_BA_12000.pth', help='path to pre-trained G_BA')
parser.add_argument('--imgNum', type=int, default=16, help='image number')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
##########   DATASET   ###########

datasetA = DATASET_fits(opt.oridataPath,opt.fineSize)
loader_A= torch.utils.data.DataLoader(dataset=datasetA,
                                       batch_size=opt.batchSize,
                                       shuffle=False,
                                       num_workers=1)
loaderA = iter(loader_A)


datasetB = DATASET_fits(opt.bludataPath,opt.fineSize)
loader_B= torch.utils.data.DataLoader(dataset=datasetB,
                                       batch_size=opt.batchSize,
                                       shuffle=False,
                                       num_workers=1)
loaderB = iter(loader_B)

###########  MODEL   ###########
ndf = opt.ndf
ngf = opt.ngf

G_AB = Generator(opt.input_nc, opt.output_nc, opt.ngf)
G_BA = Generator(opt.output_nc, opt.input_nc, opt.ngf)

if(opt.G_BA != ''):
    print('Warning! Loading pre-trained weights.')
    G_AB.load_state_dict(torch.load(opt.G_AB))
    G_BA.load_state_dict(torch.load(opt.G_BA))
    
else:
   print('ERROR! G_AB and G_BA must be provided!')

if(opt.cuda):
    G_AB.cuda()
    G_BA.cuda()

###########   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize

real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)

real_A = Variable(real_A)
real_B = Variable(real_B)

if(opt.cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()

###########   Testing    ###########
def test():

    for i in range(0,opt.imgNum,opt.batchSize):
        imgA = loaderA.next() 
        imgB = loaderB.next()
        real_A.data.copy_(imgA)
        real_B.data.copy_(imgB)
        print(i)
        LoadSaveFits.save_fit(imgA.data,'test_or%d'%i,'./out_picture/out_test/')
        LoadSaveFits.save_fit(imgB.data,'test_de%d'%i,'./out_picture/out_test/')
        BA = G_BA(real_B)
        LoadSaveFits.save_fit(BA.data,'testBA%d'%i,'./out_picture/out_test/')
        AB = G_AB(real_A)        
        LoadSaveFits.save_fit(AB.data,'testAB%d'%i,'./out_picture/out_test/')  
         
test()


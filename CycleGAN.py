from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from itertools import chain
import time
from utils.dataset import DATASET
from utils.ImagePool import ImagePool
from model.Discriminator import Discriminator
from model.Generator import Generator

from utils.fitsFun import DATASET_fits
from utils.fitsFun import LoadSaveFits
import scripts.gradient as grad
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='train pix2pix model')
parser.add_argument('--batchSize', type=int, default=2, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=9000, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay in network D, default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='facades/train/', help='path to training images')
parser.add_argument('--bludataPath',default='',help='bleu image')
parser.add_argument('--oridataPath',default='',help='ori iamg') #or
parser.add_argument('--loadSize', type=int, default=143, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=220, help='random crop image to this size')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=1, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=1, help='channel number of output image')
parser.add_argument('--G_AB', default='', help='path to pre-trained G_AB')
parser.add_argument('--G_BA', default='', help='path to pre-trained G_BA')
parser.add_argument('--D_A', default='', help='path to pre-trained D_A')
parser.add_argument('--D_B', default='', help='path to pre-trained D_B')
parser.add_argument('--save_step', type=int, default=3000, help='save interval')
parser.add_argument('--log_step', type=int, default=3000, help='log interval')
parser.add_argument('--loss_type', default='mse', help='GAN loss type, bce|mse default is negative likelihood loss')
parser.add_argument('--poolSize', type=int, default=50, help='size of buffer in lsGAN, poolSize=0 indicates not using history')
parser.add_argument('--lambda_ABA', type=float, default=10.0, help='weight of cycle loss ABA')
parser.add_argument('--lambda_BAB', type=float, default=10.0, help='weight of cycle loss BAB')
parser.add_argument('--lambda_idt', type=float, default=1.0, help='weight of style loss')

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
##########      dataset fits  #############
datasetA = DATASET_fits(opt.oridataPath,opt.fineSize)
loader_A= torch.utils.data.DataLoader(dataset=datasetA,
                                       batch_size=opt.batchSize,
                                       shuffle=True,
                                       num_workers=2)
loaderA = iter(loader_A)

datasetB = DATASET_fits(opt.bludataPath,opt.fineSize)
loader_B = torch.utils.data.DataLoader(dataset=datasetB,
                                       batch_size=opt.batchSize,
                                       shuffle=True,
                                       num_workers=2)
loaderB = iter(loader_B)

##########   DATASET   ###########
ABPool = ImagePool(opt.poolSize)
BAPool = ImagePool(opt.poolSize)

############   MODEL   ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ndf = opt.ndf
ngf = opt.ngf

D_A = Discriminator(opt.input_nc,ndf)
D_B = Discriminator(opt.output_nc,ndf)
G_AB = Generator(opt.input_nc, opt.output_nc, opt.ngf)
G_BA = Generator(opt.output_nc, opt.input_nc, opt.ngf)

from collections import OrderedDict
if(opt.G_AB != ''): 
    print('Warning! Loading pre-trained weights.')
    G_AB.load_state_dict(torch.load(opt.G_AB))
    G_BA.load_state_dict(torch.load(opt.G_BA))
else:
    print('No load pre-trained weights ')
    G_AB.apply(weights_init)
    G_BA.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)



if(opt.cuda):
    D_A.cuda()
    D_B.cuda()
    G_AB.cuda()
    G_BA.cuda()

###########   LOSS & OPTIMIZER   ##########
criterionMSE = nn.L1Loss()

if(opt.loss_type == 'bce'):
    criterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()
# chain is used to update two generators simultaneously
optimizerD_A = torch.optim.Adam(D_A.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerD_B = torch.optim.Adam(D_B.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerG = torch.optim.Adam(chain(G_AB.parameters(),G_BA.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999))

############   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize

real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
AB = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
BA = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
label = torch.FloatTensor(opt.batchSize)

real_A = Variable(real_A)
real_B = Variable(real_B)
label = Variable(label)
AB = Variable(AB)
BA = Variable(BA)

if(opt.cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    label = label.cuda()
    AB = AB.cuda()
    BA = BA.cuda()
    criterion.cuda()
    criterionMSE.cuda()

real_label = 1
fake_label = 0

###########   Testing    ###########
def test(niter):
    loaderA, loaderB = iter(loader_A), iter(loader_B)
    imgA = loaderA.next()
    imgB = loaderB.next()
    real_A.data.resize_(imgA.size()).copy_(imgA)
    real_B.data.resize_(imgB.size()).copy_(imgB)
    AB = G_AB(real_A)
    BA = G_BA(real_B)
    LoadSaveFits.save_fit(AB.data,'AB_niter_%03d_1'%niter,'./out_picture/')
    LoadSaveFits.save_fit(BA.data,'BA_niter_%03d_1'%niter,'./out_picture/')
    LoadSaveFits.save_fit(real_A.data,'realA_niter_%03d_1'%niter,'./out_picture/')
    LoadSaveFits.save_fit(real_B.data,'realB_niter_%03d_1'%niter,'./out_picture/')    
    


###########   Training   ###########
D_A.train()
D_B.train()
G_AB.train()
G_BA.train()
plt_D = []
plt_G = []
plt_MSE = []
plt_idt = []
plt_tv = []
d=g=m=i=t=e=0
plt_error = []
s = time.time()
for iteration in range(1,opt.niter+1):
    ###########   data  ###########
    try:
        imgA = loaderA.next()
        imgB = loaderB.next()
    except StopIteration:
        loaderA, loaderB = iter(loader_A), iter(loader_B)
        imgA = loaderA.next()
        imgB = loaderB.next()

    real_A.data.resize_(imgA.size()).copy_(imgA)
    real_B.data.resize_(imgB.size()).copy_(imgB)

    ###########   fDx   ###########
    D_A.zero_grad()
    D_B.zero_grad()

    # train with real
    outA = D_A(real_A)
    outB = D_B(real_B)
  
    label.data.resize_(outA.size())
    label.data.fill_(real_label)
    l_A = criterion(outA, label)
    l_B = criterion(outB, label)
    errD_real = l_A + l_B
    errD_real.backward()

    # train with fake
    label.data.fill_(fake_label)
    AB_tmp = G_AB(real_A)

    AB.data.resize_(AB_tmp.data.size()).copy_(ABPool.Query(AB_tmp.cpu().data))
    BA_tmp = G_BA(real_B)
    BA.data.resize_(BA_tmp.data.size()).copy_(BAPool.Query(BA_tmp.cpu().data))
    
    out_BA = D_A(BA.detach())
    out_AB = D_B(AB.detach())

    l_BA = criterion(out_BA,label)
    l_AB = criterion(out_AB,label)

    errD_fake = l_BA + l_AB
    errD_fake.backward()

    errD = (errD_real + errD_fake)*0.5
    optimizerD_A.step()
    optimizerD_B.step()

    ########### fGx ###########
    G_AB.zero_grad()
    G_BA.zero_grad()
    label.data.fill_(real_label)

    AB = G_AB(real_A)
    ABA = G_BA(AB)

    BA = G_BA(real_B)
    BAB = G_AB(BA)

    out_BA = D_A(BA)
    out_AB = D_B(AB)

    l_BA = criterion(out_BA,label)
    l_AB = criterion(out_AB,label)
    #TV loss
    l_TV_BAB = grad.gradient(BAB.detach())
    l_TV_BAB = l_TV_BAB.cuda()
    l_TV_ABA = grad.gradient(ABA.detach()) 
    l_TV_ABA = l_TV_ABA.cuda()
    errTV  = l_TV_ABA + l_TV_BAB

    # identity loss
    l_idt = opt.lambda_idt*criterionMSE(AB,real_A) + opt.lambda_idt*criterionMSE(BA,real_B)

    # reconstruction loss
    l_rec_ABA = criterionMSE(ABA, real_A) * opt.lambda_ABA
    l_rec_BAB = criterionMSE(BAB, real_B) * opt.lambda_BAB 

    errGAN = l_BA + l_AB
    errMSE = l_rec_ABA + l_rec_BAB
    errG = errGAN + errMSE + l_idt + errTV  
    errG.backward()

    optimizerG.step()

    ###########   Logging   ############
    if(iteration % opt.niter):
        print('[%d/%d]L_all:%.2f Loss_D:%.2f Loss_G:%.2f Loss_MSE:%.2f Loss_idt:%.2f Loss_TV:%.2f'
                  %(iteration, opt.niter,errG.data[0],errD.data[0], errGAN.data[0], errMSE.data[0],l_idt.data[0],errTV.data[0]))
    ########## Visualize #########
    if iteration % opt.save_step == 0:
        test(iteration)

    if iteration % opt.save_step == 0:
        torch.save(G_AB.state_dict(), '{}/G_AB_{}.pth'.format(opt.outf, iteration))
        torch.save(G_BA.state_dict(), '{}/G_BA_{}.pth'.format(opt.outf, iteration))
        torch.save(D_A.state_dict(), '{}/D_A_{}.pth'.format(opt.outf, iteration))
        torch.save(D_B.state_dict(), '{}/D_B_{}.pth'.format(opt.outf, iteration))

    e += errG.data[0]
    d += errD.data[0]
    g += errGAN.data[0]
    m += errMSE.data[0]
    i += l_idt.data[0]
    t += errTV.data[0]
    count = 100
    if (iteration%count==0):
        e /= count
        d /= count
        g /= count
        m /= count
        i /= count
        t /= count
        plt_error.append(e)
        plt_D.append(d)
        plt_G.append(g)
        plt_MSE.append(m)
        plt_idt.append(i)
        plt_tv.append(t)
        d=g=m=i=t=e=0
    end = time.time()
print (iteration,'time %f (s)'%(end -s))
x = np.linspace(1,len(plt_error),len(plt_error))  
plt.figure()
plt.subplot(1,1,1)
y = plt_error
plt.plot(x, y,'r',label='errall')# use pylab to plot x and y
y = plt_D
plt.plot(x, y,'b',label='errD')
y = plt_G
plt.plot(x, y,'c',label='errGAN')
y = plt_MSE
plt.plot(x, y,'m',label='errMSE_cycle')
y = plt_idt
plt.plot(x, y,'g',label='errIdt')
y = plt_tv
plt.plot(x, y,'k',label='errTV')
plt.title('loss function')# give plot a title
plt.legend()
plt.savefig('./out_picture/plot.eps') 
print('finish')








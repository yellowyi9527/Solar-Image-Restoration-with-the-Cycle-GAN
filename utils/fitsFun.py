import PIL.Image as Image
from astropy.io import fits
import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.utils.data as Data

class LoadSaveFits:
    def __init__(self,path,img,name):
        self.path = path
        self.img = img
        self.name = name
        
    def norm(img):
        img = (img - np.min(img))/(np.max(img) - np.min(img)) #normalization
        img -= np.mean(img)  # take the mean
        img /= np.std(img)  #standardization
        img = np.array(img,dtype='float32')
        return img
    
    def read_fits(path):
        hdu = fits.open(path)
        img = hdu[0].data
        img = np.array(img,dtype = np.float32)
        hdu.close()
        return img
    
    def save_fit_cpu(img,name,path):
        if os.path.exists(path + name+'.fits'):
            os.remove(path + name+'.fits')
        grey=fits.PrimaryHDU(img)
        greyHDU=fits.HDUList([grey])
        greyHDU.writeto(path + name+'.fits')
        
    def save_fit(img,name,path):
        if torch.cuda.is_available(): 
            img = torch.Tensor.cpu(img)
            img = img.data.numpy()
        else:
            img = np.array(img)
        if os.path.exists(path + name+'.fits'):
            os.remove(path + name+'.fits')
        grey=fits.PrimaryHDU(img)
        greyHDU=fits.HDUList([grey])
        greyHDU.writeto(path + name+'.fits')

# load data of cycleGAN
class DATASET_fits():
    def __init__(self,dataPath='',fineSize=512):
        super(DATASET_fits, self).__init__()
        # list all images into a list
        self.list = [x for x in os.listdir(dataPath)]
        self.dataPath = dataPath
        self.fineSize = fineSize

        
    def __getitem__(self, index):
        path = os.path.join(self.dataPath,self.list[index])
        img = LoadSaveFits.read_fits(path) 
        h,w = img.shape
        img = img[int((h/2-self.fineSize/2)):int((h/2+self.fineSize/2)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        img = LoadSaveFits.norm(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
#        img = torch.cat((img,img,img),0) 
        return img
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.list)


 # load data of cycleGAN
class DATASET_fits_sort():
    def __init__(self,dataPath='',fineSize=512):
        super(DATASET_fits_sort, self).__init__()
        # list all images into a list
        namefile = os.listdir(dataPath)
        namefile.sort(key=lambda x: int(x[3:-5]))
        print(namefile)
        self.list = [x for x in namefile]
        self.dataPath = dataPath
        self.fineSize = fineSize

        
    def __getitem__(self, index):
        path = os.path.join(self.dataPath,self.list[index])
        img = LoadSaveFits.read_fits(path) 
        h,w = img.shape
        bias = 200
        img = img[int((h/2-self.fineSize/2+bias)):int((h/2+self.fineSize/2+bias)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        # a= 200
        # b= 200
        # img= img[a:a+256,b:b+256]
        img = LoadSaveFits.norm(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
#        img = torch.cat((img,img,img),0) 
        return img
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.list)       



class DATASET_fits_sort2():
    def __init__(self,dataPath='',fineSize=512):
        super(DATASET_fits_sort2, self).__init__()
        # list all images into a list
        namefile = os.listdir(dataPath)
        namefile.sort(key=lambda x: int(x[0:-5]))
        print(namefile)
        self.list = [x for x in namefile]
        self.dataPath = dataPath
        self.fineSize = fineSize

        
    def __getitem__(self, index):
        path = os.path.join(self.dataPath,self.list[index])
        img = LoadSaveFits.read_fits(path) 
        h,w = img.shape
        img = img[int((h/2-self.fineSize/2)):int((h/2+self.fineSize/2)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        img = LoadSaveFits.norm(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
#        img = torch.cat((img,img,img),0) 
        return img
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.list)   

        
#load data of pix2pix
class DATASET_fits2():
    def __init__(self,dataPath_or= '' , dataPath_de='',fineSize=512):
        super(DATASET_fits2,self).__init__()
        self.list = [x for x in os.listdir(dataPath_or)]
        self.dataPath_or = dataPath_or
        self.dataPath_de =  dataPath_de
        self.fineSize = fineSize
    
    def __getitem__(self,index):
        path_or = os.path.join(self.dataPath_or,self.list[index])
        path_de = os.path.join(self.dataPath_de,self.list[index])
        img_or = LoadSaveFits.read_fits(path_or)
        img_de = LoadSaveFits.read_fits(path_de)
        h,w = img_de.shape
        img_or = img_or[int((h/2-self.fineSize/2)):int((h/2+self.fineSize/2)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        img_or = LoadSaveFits.norm(img_or)
        img_de = img_de[int((h/2-self.fineSize/2)):int((h/2+self.fineSize/2)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        img_de = LoadSaveFits.norm(img_de)
        img_or = torch.from_numpy(img_or)
        img_or = img_or.unsqueeze(0)
        img_de = torch.from_numpy(img_de)
        img_de = img_de.unsqueeze(0)
        return img_or,img_de
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.list)
        
        

    




    

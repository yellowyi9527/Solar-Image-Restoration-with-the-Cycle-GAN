#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:42:40 2018

@author: yellow
"""

import cv2 
import numpy as np
import torch
def gradient(img):
    img = img.squeeze(1)
    a,_,_ = img.shape
    img = np.array(img)
    mean = 0
    for i in range(0,a):
        img1 = img[i]
        Ax = cv2.Scharr(img1,cv2.CV_16S,1,0)
        Ay = cv2.Scharr(img1,cv2.CV_16S,0,1)
        Adst = np.abs(Ax) + np.abs(Ay)
        mean += np.mean(Adst)
    return torch.tensor(mean/a)
        
    
    
    
    

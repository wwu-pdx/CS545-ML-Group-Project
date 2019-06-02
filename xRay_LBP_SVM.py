# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:17:12 2019

@author: Wen
"""

## Source code from: https://scottontechnology.com/open-multiple-images-opencv-python/
# import the necessary packages
import cv2
import os, os.path
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
#import itertools

#LBP params
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

#image path and valid extensions
def ProcessGreyImgs(dirc):
    imageDir = dirc #specify your path here
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]
     
    #create a list all files in directory and
    #append files with a vaild extention to image_path_list
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))
     #end
    images = []
    #loop through image_path_list to open each image
    for imagePath in image_path_list:
        image = cv2.imread(imagePath,0)#0 grey image
        # image = cv2.imread(imagePath) 
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        image=ReSize(image)
        image=LocalBinaryPattern(image)
        image=Histogram(image)
        #FitTransform(image)
        images.append(image)
        
#    for i in range(5):
#        image = cv2.imread(image_path_list[i],0)
#        image=ReSize(image)
#        image=LocalBinaryPattern(image)
#        image=Histogram(image)
#        FitTransform(image)
#        images.append(image)
#    end
    return np.squeeze(np.asarray(images),axis=2)
#end

def  ReSize(img):   
    desired_size = 50
    old_size = img.shape # old_size is in (height, width) format    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])   
    # new_size should be in (height, width) format
    im = cv2.resize(img, (new_size[1], new_size[0]))
    return im

def LocalBinaryPattern(img):
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    return lbp

def Histogram(imgf):
    n_bins =n_points+2 ##int(imgf.max() + 1)
    hist  = cv2.calcHist([imgf.astype('float32')],[0],None,[n_bins],[0,n_bins])
    cv2.normalize(hist,hist)
    return hist

def FitTransform(f):
    scaler = StandardScaler()   
    rescaled_features = scaler.fit_transform(f)
    return rescaled_features
    
#def SVCPredict():
train_n=ProcessGreyImgs("/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL")
target_n=[0]*train_n.shape[0]
train_p=ProcessGreyImgs("/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA")
target_p=[1]*train_p.shape[0]
trainSet=np.concatenate((train_n,train_p), axis=0)
targets=target_n+target_p

test_n=ProcessGreyImgs("/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL")
test_p=ProcessGreyImgs("/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA")



clf = svm.SVC(gamma='scale')
clf.fit(trainSet, np.asarray(targets))
predict_test_n=clf.predict(test_n)
predict_test_p=clf.predict(test_p)

accuraccyn=100.00-np.sum(predict_test_n)*100.00/test_n.shape[0]#normal
accuraccyp=np.sum(predict_test_p)*100.00/test_p.shape[0]#pneumonia


    
print("done")
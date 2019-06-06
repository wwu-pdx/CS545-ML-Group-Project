#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:24:26 2019

@author: jay
"""
# import the necessary packages
import cv2
import os, os.path
import numpy as np
 
def resizeSquare(imageList, finalDim):
    """
    Inputs: 
        imageList: list of greyscale image arrays (cv2.imread)
        finalDim: length of resized image (square: finalDim x finalDim)
    Output:
        list of greyscale image arrays rescaled with aspect ratio preserved
        black pixels replace "empty space" in the image
    Source: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    changed to greyscale, iterate through a list
    """
    newImageList = []
    for myImage in imageList:
        old_size = myImage.shape[:2] # old_size is in (height, width) format
        ratio = float(finalDim)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # new_size should be in (width, height) format
        im = cv2.resize(myImage, (new_size[1], new_size[0]))
        delta_w = finalDim - new_size[1]
        delta_h = finalDim - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = 0
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        newImageList.append(new_im)
    return newImageList

def readImageDir(myPath, myLabel):
    """
    Input: myPath as a string, myLabel 
    Source code from: https://scottontechnology.com/open-multiple-images-opencv-python/
    """
#image path and valid extensions
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]
     
    #create a list all files in directory and
    #append files with a vaild extention to image_path_list
    for file in os.listdir(myPath):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(myPath, file))
     
    imageList = []
    #loop through image_path_list to open each image
    for imagePath in image_path_list:
        image = cv2.imread(imagePath, 0)
        imageList.append(image)
    
    labelList = [myLabel]*len(imageList)
    return (imageList, labelList)

resizedDim = 250

## Training data: normal
trainNPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/train/NORMAL/" #specify your path here
trainNRawImageL, trainNLabelL = readImageDir(trainNPath, 0)
trainNResizedImageL = resizeSquare(trainNRawImageL, resizedDim)

## Training data: pneumonia 
trainPPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/train/PNEUMONIA/" #specify your path here
trainPRawImageL, trainPLabelL = readImageDir(trainPPath, 1)
trainPResizedImageL = resizeSquare(trainPRawImageL, resizedDim)

## Merge the training sets together
## Normal will be index 0-1340. Pneumonia will be 1341-2682
trainResizedImageList = trainNResizedImageL + trainPResizedImageL
trainLabels = trainNLabelL + trainPLabelL

## Run histogram equalization (adaptive)
## source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
## adapative (local) histogram equalization
## create a CLAHE object (Arguments are optional).
trainResizedEqualizedImages = []
for image in trainResizedImageList:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    cl1 = clahe.apply(image)
    trainResizedEqualizedImages.append(cl1)
    
## show an example of the equalization before-after
cv2.imshow("local equalize", trainResizedEqualizedImages[2200])
cv2.imshow("original", trainResizedImageList[2200])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:24:26 2019

@author: Colleen Xu and Hoi Dinh
"""
# import the necessary packages
import cv2
import os, os.path
import numpy as np 
import skimage.feature as skif
from sklearn.utils import shuffle
from sklearn import svm, preprocessing
from sklearn.model_selection import GridSearchCV
import pandas as pd
import sklearn.metrics as skmetrics

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

## length of resized image (square)
resizedDim = 250

## Training data: normal. label "normal" as 0. 
trainNPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/train/NORMAL/" #specify your path here
trainNRaw, trainNLabel = readImageDir(trainNPath, 0)
trainNResized = resizeSquare(trainNRaw, resizedDim)

## Training data: pneumonia. label "pneumonia" as 1.
trainPPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/train/PNEUMONIA/" #specify your path here
trainPRaw, trainPLabel = readImageDir(trainPPath, 1)
trainPResized = resizeSquare(trainPRaw, resizedDim)

## TRUNCATE Pneumonia training samples to make it balanced.
## first shuffle to make what we truncate random
trainPResized = shuffle(trainPResized, random_state=0)
trainPResized = trainPResized[0:len(trainNResized)]
trainPLabel = trainPLabel[0:len(trainPResized)]

## Merge the training sets together
## Normal will be index 0-1340. Pneumonia will be 1341-5216
trainResized = trainNResized + trainPResized
trainLabels = trainNLabel + trainPLabel

## Run histogram equalization (adaptive)
## source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
## adapative (local) histogram equalization
## create a CLAHE object (Arguments are optional).
trainEqualized = []
for image in trainResized:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    cl1 = clahe.apply(image)
    trainEqualized.append(cl1)

## HOG trial:  
## Source: https://stackoverflow.com/a/42059758, adapted to iterate through list of images
#trial = skif.hog(trainEqualized[2110], visualize=True, pixels_per_cell=(50,50), cells_per_block=(2,2))
#trial[0].shape
### show an example of the equalization before-after
#cv2.imshow("HOG", trial[1])
#cv2.imshow("original", trainResized[2110])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1)

trainHOGFeats = []
idx = 0  ## used to watch it run

for image in trainEqualized:  ## use trainResized if not using equalized stuff
    idx +=1 
    if (idx % 10)==0:
        print("Still running ", idx)  ## used to mark progress of run
    ## first tried pixels_per_cell of 50x50, cells per block of 2x2
    ## second: tried pixels per cell 25x25, cells per block 3x3
    hog = skif.hog(image,pixels_per_cell=(50,50), cells_per_block=(3,3))
    trainHOGFeats.append(hog)
    
## SVM prep!
## convert list to 2-D array for SVM: end up with 5216 rows x 9216 columns
trainHOGFeats = np.array(trainHOGFeats)
## convert labels list to numpy array
trainLabels = np.array(trainLabels)
## randomly shuffle HOG Feats and labels together
trainHOGFeats, trainLabels =  shuffle(trainHOGFeats, trainLabels, random_state=0)
## scale data. It will be able to reapply the same transformation to the testing set
## reference: https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
scalerHOG = preprocessing.StandardScaler().fit(trainHOGFeats)
trainHOGFeats = scalerHOG.transform(trainHOGFeats)

## SVM time! Picking the best kernels, hyperparameters
## Commented out since it can take a while to run 
## Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
#parameters = {'kernel':('rbf',), 'C':[0.1, 1, 10, 100],'gamma':['scale', 0.01, 0.001, 0.0001]}
#svcHOG = svm.SVC()
#clfHOG = GridSearchCV(svcHOG, parameters, cv=5, verbose=2)
#clfHOG.fit(trainHOGFeats, trainLabels)
## for pixels_per_cell=(50,50), cells_per_block=(2,2)
## best was C=10, gamma='scale', kernel='rbf' with mean score of 0.9748849693251533

## for pixels_per_cell=(25,25), cells_per_block=(3,3)
## best was C=100, gamma=0.0001, kernel=rbf

## for non-equalized, pixels_per_cell=(50,50), cells_per_block=(3,3)
## C=10, gamma=0.001, kernel='rbf'

## params for balanced dataset. Ran CV ONLY on RBF
HOG_SVC = svm.SVC(C=10, kernel='rbf', gamma=0.001)
HOG_SVC.fit(trainHOGFeats, trainLabels)

## get test data to predict on. label normal as 0, pneumonia as 1
testNPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/test/NORMAL/" #specify your path here
testNRaw, testNLabel = readImageDir(testNPath, 0)
testNResized = resizeSquare(testNRaw, resizedDim)
testPPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/test/PNEUMONIA/" #specify your path here
testPRaw, testPLabel = readImageDir(testPPath, 1)
testPResized = resizeSquare(testPRaw, resizedDim)
## Merge the test sets together: get 624 total, 234 normal and 390 pneumonia
testResized = testNResized + testPResized
testLabels = testNLabel + testPLabel
## Run histogram equalization (adaptive)
## source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
## adapative (local) histogram equalization
testEqualized = []
for image in testResized:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    cl1 = clahe.apply(image)
    testEqualized.append(cl1)
## GLCM trial: with 24 features  
## Source: https://stackoverflow.com/a/42059758, adapted to iterate through list of images
testHOGFeats = []
idx = 0  ## used to watch it run
for image in testEqualized:  ## ## use testResized if not using equalized stuff
    idx +=1 
    if (idx % 10)==0:
        print("Still running ", idx)  ## used to mark progress of run
    hog = skif.hog(image,pixels_per_cell=(50,50), cells_per_block=(3,3))
    testHOGFeats.append(hog)

## SVM prep!
## convert list to 2-D array for SVM: end up with 624 rows x 24 columns
testHOGFeats = np.array(testHOGFeats)
## convert labels list to numpy array
testLabels = np.array(testLabels)
## randomly shuffle GLCM Feats and labels together
testHOGFeats, testLabels =  shuffle(testHOGFeats, testLabels, random_state=0)
## scale testGLCMFeats the same way the training data was scaled
testHOGFeats = scalerHOG.transform(testHOGFeats)

## Predict using trained SVM
## confusion matrix
## reference: https://datatofish.com/confusion-matrix-python/
## get predictions for all 624 test samples      
HOGPredictions = HOG_SVC.predict(testHOGFeats)

finalData = {'Predicted':HOGPredictions, 'Truth':testLabels}
finalDF = pd.DataFrame(finalData, columns=['Predicted', 'Truth'])
conMat = pd.crosstab(finalDF['Predicted'], finalDF['Truth'], rownames=['Predicted'], colnames=['Truth'])
print(conMat)
#skmetrics.confusion_matrix(testLabels, GLCMPredictions)
## can do further metrics, like accuracy, ROC, etc. 
print("accuracy: ", skmetrics.accuracy_score(testLabels, HOGPredictions))
print("precision: ", skmetrics.precision_score(testLabels, HOGPredictions))
print("recall: ", skmetrics.recall_score(testLabels, HOGPredictions))
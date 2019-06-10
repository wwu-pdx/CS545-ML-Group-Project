# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Sean Lesch and Colleen Xu
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
print("Reading train normal data")
trainNRaw, trainNLabel = readImageDir(trainNPath, 0)
trainNResized = resizeSquare(trainNRaw, resizedDim)
# To free up space, we need to remove these large objects frequently. Otherwise
# the program will use 12 GB of memory.
del trainNRaw
print("Reading train pneumonia data")
## Training data: pneumonia. label "pneumonia" as 1.
trainPPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/train/PNEUMONIA/" #specify your path here
trainPRaw, trainPLabel = readImageDir(trainPPath, 1)
trainPResized = resizeSquare(trainPRaw, resizedDim)
del trainPRaw

## TRUNCATE Pneumonia training samples to make it balanced.
## first shuffle to make what we truncate random
trainPResized = shuffle(trainPResized, random_state=0)
trainPResized = trainPResized[0:len(trainNResized)]
trainPLabel = trainPLabel[0:len(trainPResized)]

## Merge the training sets together
## Normal will be index 0-1340. Pneumonia will be 1341-2682
trainResized = trainNResized + trainPResized
trainLabels = trainNLabel + trainPLabel
del trainPResized
del trainNResized

## Run histogram equalization (adaptive)
## source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
## adapative (local) histogram equalization
## create a CLAHE object (Arguments are optional).
trainEqualized = []
for image in trainResized:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    cl1 = clahe.apply(image)
    trainEqualized.append(cl1)
    
del trainResized

## GLCM trial: with 24 features  
## Source: https://stackoverflow.com/a/42059758, adapted to iterate through list of images
trainGLCMFeats = []
distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, (np.pi*3)/4]
properties = ['energy', 'homogeneity', 'contrast', 'dissimilarity', 'correlation', 'ASM']
idx = 0  ## used to watch it run

## preserve metrics across all distances. 
# image = trainEqualized[9]
for image in trainEqualized:
    idx +=1 
    if (idx % 10)==0:
        print("Still running ", idx)  ## used to mark progress of run
    glcm = skif.greycomatrix(image, distances=distances, angles=angles,symmetric=True,normed=True)
    finalFeats = []
    for prop in properties:
        means = np.mean(skif.greycoprops(glcm, prop), axis=1)
        finalFeats.append(means)
    finalFeats = np.array(finalFeats).flatten()
    #feats = np.hstack([skif.greycoprops(glcm, prop).ravel() for prop in properties])
    trainGLCMFeats.append(finalFeats)
    
## SVM prep!
## convert list to 2-D array for SVM: end up with 5216 rows x 24 columns
trainGLCMFeats = np.array(trainGLCMFeats)
## convert labels list to numpy array
trainLabels = np.array(trainLabels)
## randomly shuffle GLCM Feats and labels together
trainGLCMFeats, trainLabels =  shuffle(trainGLCMFeats, trainLabels, random_state=0)

## scale data. It will be able to reapply the same transformation to the testing set
## reference: https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
scaler = preprocessing.StandardScaler().fit(trainGLCMFeats)
trainGLCMFeats = scaler.transform(trainGLCMFeats)

## SVM time! Picking the best kernels, hyperparameters
## Commented out since it can take a while to run 
## Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
#parameters = {'kernel':('rbf',), 'C':[0.1, 1, 10, 100],'gamma':['scale', 0.01, 0.001, 0.0001]}
#svc = svm.SVC()
#clf = GridSearchCV(svc, parameters, cv=5, verbose=2)
#clf.fit(trainGLCMFeats, trainLabels)
## distance=1 result was that the best was C=100, gamma='scale', kernel='rbf'
## had best mean score of 0.9321319018404908
## distance=1,2 result was that the best was C=100, gamma=.01, kernel='rbf'
## Distance = [1,2,3] and using mean of GLCM
##{'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}
## Predicted  
'''        
0           88   10
1          146  380
accuracy:  0.75
precision:  0.7224334600760456
recall:  0.9743589743589743
'''

## Using GLCM mean for feature extraction distance [1,2]
##(C=100, kernel='rbf', gamma='scale')
'''
Predicted          
0           85   17
1          149  373
accuracy:  0.7339743589743589
precision:  0.7145593869731801
recall:  0.9564102564102565
'''
## Distance = [1,2,3] and using mean of GLCM
##{'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}
'''
Truth        0    1
Predicted          
0           91   22
1          143  368
accuracy:  0.7355769230769231
precision:  0.7201565557729941
recall:  0.9435897435897436
'''


## Distance = [1,2,3] and using mean of GLCM
## using truncated dataset: {'C': 100, 'gamma': 'scale', 'kernel': 'linear'}
## But we chose to use 'rbf': ran with best params from 'rbf'
GLCM_SVC = svm.SVC(C=100, kernel='rbf', gamma='scale')
GLCM_SVC.fit(trainGLCMFeats, trainLabels)

## get test data to predict on. label normal as 0, pneumonia as 1
testNPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/test/NORMAL/" #specify your path here
testNRaw, testNLabel = readImageDir(testNPath, 0)
testNResized = resizeSquare(testNRaw, resizedDim)
del testNRaw
testPPath = "/Users/jay/Desktop/PSUMachineLearn/chest_xray/test/PNEUMONIA/" #specify your path here
testPRaw, testPLabel = readImageDir(testPPath, 1)
testPResized = resizeSquare(testPRaw, resizedDim)
del testPRaw
## Merge the test sets together: get 624 total, 234 normal and 390 pneumonia
testResized = testNResized + testPResized
testLabels = testNLabel + testPLabel
del testNResized
del testPResized
## Run histogram equalization (adaptive)
## source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
## adapative (local) histogram equalization

testEqualized = []
for image in testResized:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    cl1 = clahe.apply(image)
    testEqualized.append(cl1)

del testResized

## GLCM trial: with 24 features  
## Source: https://stackoverflow.com/a/42059758, adapted to iterate through list of images
testGLCMFeats = []
idx = 0  ## used to watch it run

# image = trainEqualized[9]
for image in testEqualized:
    idx +=1 
    if (idx % 10)==0:
        print("Still running ", idx)  ## used to mark progress of run
    glcm = skif.greycomatrix(image, distances=distances, angles=angles,symmetric=True,normed=True)
    finalFeats = []
    for prop in properties:
        means = np.mean(skif.greycoprops(glcm, prop), axis=1)
        finalFeats.append(means)
    finalFeats = np.array(finalFeats).flatten()
    #feats = np.hstack([skif.greycoprops(glcm, prop).ravel() for prop in properties])
    testGLCMFeats.append(finalFeats)
    

## SVM prep!
## convert list to 2-D array for SVM: end up with 624 rows x 24 columns
testGLCMFeats = np.array(testGLCMFeats)
## convert labels list to numpy array
testLabels = np.array(testLabels)
## randomly shuffle GLCM Feats and labels together
testGLCMFeats, testLabels =  shuffle(testGLCMFeats, testLabels, random_state=0)
## scale testGLCMFeats the same way the training data was scaled
testGLCMFeats = scaler.transform(testGLCMFeats)

## Predict using trained SVM
## confusion matrix
## get predictions for all 624 test samples      
GLCMPredictions = GLCM_SVC.predict(testGLCMFeats)

finalData = {'Predicted':GLCMPredictions, 'Truth':testLabels}
finalDF = pd.DataFrame(finalData, columns=['Predicted', 'Truth'])
conMat = pd.crosstab(finalDF['Predicted'], finalDF['Truth'], rownames=['Predicted'], colnames=['Truth'])
print(conMat)
#skmetrics.confusion_matrix(testLabels, GLCMPredictions)
## can do further metrics, like accuracy, ROC, etc. 
print("accuracy: ", skmetrics.accuracy_score(testLabels, GLCMPredictions))
print("precision: ", skmetrics.precision_score(testLabels, GLCMPredictions))
print("recall: ", skmetrics.recall_score(testLabels, GLCMPredictions))

## show an example of the equalization before-after
# cv2.imshow("local equalize", trainResizedEqualizedImages[2110])
#cv2.imshow("original", trainResized[999])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1)
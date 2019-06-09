# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmetrics
import pandas as pd

#LBP params
radius = 3
n_points = 100 * radius
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
        image=ReSize(image)
        image=TrainEqualized(image)
        image=LocalBinaryPattern(image)
        image=Histogram(image)
        #FitTransform(image)
        images.append(image)
#        
#    for i in range(5):
#        image = cv2.imread(image_path_list[i],0)
#        image=ReSize(image)
#        image=LocalBinaryPattern(image).flatten()
#        image=Histogram(image)
#        #FitTransform(image)
#        images.append(image)
    #end
    #return np.squeeze(np.asarray(images),axis=2)
    return np.asarray(images)
#end

def  ReSize(img):   
    desired_size = 250
    old_size = img.shape # old_size is in (height, width) format    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])   
    # new_size should be in (height, width) format
    im = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = 0
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im


#Source https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
def LocalBinaryPattern(img):
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
#    cv2.normalize(lbp,lbp)
    return lbp
def Histogram(imgf):
    eps=1e-7
    n_bins =n_points+2 ##int(imgf.max() + 1)
    #hist  = cv2.calcHist([imgf.astype('float32')],[0],None,[n_bins],[0,n_bins])
    hist, _ = np.histogram(imgf.ravel(), bins=range(0,n_bins+1), range=(0, n_bins))
#    cv2.normalize(hist,hist)
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

#def FitTransform(f):
#    scaler = StandardScaler()   
#    rescaled_features = scaler.fit_transform(f)
#    return rescaled_features

def TrainEqualized(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    cl1 = clahe.apply(image)
    return cl1


#prepare train data
print("start" +datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
train_n=ProcessGreyImgs("/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL")
target_n=[0]*train_n.shape[0]
print("done train normal "+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
train_p=ProcessGreyImgs("/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA")
target_p=[1]*train_p.shape[0]
print("done train pneumonia "+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
trainSet=np.concatenate((train_n,train_p), axis=0)
targets=target_n+target_p
trainSet, targets = shuffle(trainSet, targets, random_state=0)
## reference: https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
scaler = StandardScaler().fit(trainSet)
trainSet = scaler.transform(trainSet)



#prepare test data
print("done train all " +datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
test_n=ProcessGreyImgs("/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL")
print("done test normal "+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
test_p=ProcessGreyImgs("/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA")
print("done test pneumonia "+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
testLabels_n=[0]*test_n.shape[0]
testLabels_p=[1]*test_p.shape[0]
testLabels=testLabels_n+testLabels_p
testSet=np.concatenate((test_n,test_p), axis=0)
testSet, testLabels = shuffle(testSet, testLabels, random_state=0)
#scale
testSet = scaler.transform(testSet)


## Commented out since it can take a while to run 
## Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
#parameters = {'kernel':('rbf',), 'C':[0.1, 1, 10, 100],'gamma':['scale', 0.01, 0.001, 0.0001]}
#svc = svm.SVC()
#clfLBP = GridSearchCV(svc, parameters, cv=5, verbose=2)
#clfLBP.fit(trainSet, targets)
## current LBP result was that the best was C=100, gamma=0.01, kernel='rbf'
## had best mean score of 0.9229294478527608
## best para for lbp rbf, 100, 0.01

#fit in SVM and predict
clf = svm.SVC(C=100, kernel='rbf', gamma=0.01)
#clf = svm.SVC( gamma='scale')
clf.fit(trainSet, np.asarray(targets))



print("done fit "+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
LBPPredictions=clf.predict(testSet)
print("done predict  "+datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

#confusion matrix
finalData = {'Predicted':LBPPredictions, 'Truth':testLabels}
finalDF = pd.DataFrame(finalData, columns=['Predicted', 'Truth'])
conMat = pd.crosstab(finalDF['Predicted'], finalDF['Truth'], rownames=['Predicted'], colnames=['Truth'])
print(conMat)


#skmetrics.confusion_matrix(testLabels, GLCMPredictions)
## can do further metrics, like accuracy, ROC, etc. 
print("accuracy: ", skmetrics.accuracy_score(testLabels, LBPPredictions))
print("precision: ", skmetrics.precision_score(testLabels, LBPPredictions))
print("recall: ", skmetrics.recall_score(testLabels, LBPPredictions))


   
print("done")
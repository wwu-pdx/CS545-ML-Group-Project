# CS545-ML-Group-Project

This project investigated the effects of different feature extractions on training Support Vector Machines (SVM) to classify chest X-rays as showing signs of pneumonia. 
The feature extraction methods used were
1.Grey Level Co-occurrence Matrices (GLCM)
2.Histogram of Oriented Gradients (HOG)
3.Local Binary Pattern Histograms (LBP)
For these feature extraction methods, we cross-validated SVM models to find hyperparameters that gave the best mean score to conduct training and testing using the feature extraction methods.
For each feature extraction method, the confusion matrix, accuracy, precision and recall were computed and compared to determine the differences between them in terms of their classification performance. 

Dataset for this project can be downloaded from here:

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

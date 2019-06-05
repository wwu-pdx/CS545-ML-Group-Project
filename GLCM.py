import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops

sarraster = np.array(new_im)


#Modified from https://stackoverflow.com/questions/35551249/implementing-glcm-texture-feature-with-scikit-image-and-python
#user2341961's code for satellite imagery

#sarraster is original image, testraster will receive texture
testraster = np.copy(sarraster)
testraster[:] = 0

for i in range(testraster.shape[0] ):
    #print(time.time())
    for j in range(testraster.shape[1] ):

        #windows needs to fit completely in image
        if i <3 or j <3:
            continue
        if i > (testraster.shape[0] - 4) or j > (testraster.shape[0] - 4):
            continue

        #Calculate GLCM on a 7x7 window
        glcm_window = sarraster[i-3: i+4, j-3 : j+4]
        glcm = greycomatrix(glcm_window, [1], [0])

        #Calculate contrast and replace center pixel
        contrast = greycoprops(glcm, 'contrast')
        testraster[i,j]= contrast

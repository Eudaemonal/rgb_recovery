#!/usr/bin/python3.5


import numpy as np
import cv2


img = cv2.imread('./DataSamples/s5.jpg',0)


# Variable init
height, width = img.shape 
h = int(height/3)
w = width


# Create array of RGB values
im_g = img[0 : h, 0:w] 
im_b = img[h : 2*h, 0:w]
im_r = img[2*h : 3*h, 0:w]

lower_reso = cv2.pyrDown(im_g)

cv2.imshow("high", im_g)
cv2.imshow("low", lower_reso)
cv2.waitKey(0)

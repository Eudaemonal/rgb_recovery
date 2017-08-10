#!/usr/bin/python3.5

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt



img = cv2.imread('./DataSamples/s5.jpg',0)


# Variable init
height, width = img.shape 
h = int(height/3)
w = width
movement = 20
crop = 20
w_c = int(w/2)
h_c = int(h/2)


sample = img[h_c - crop :h_c + crop, w_c - crop:w_c + crop]
ret, imgf = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print(ret)
mid = int(ret)



# Create array of RGB values
im_g = img[0 : h, 0:w] 
im_b = img[h : 2*h, 0:w]
im_r = img[2*h : 3*h, 0:w]


min_d =0
min_i = 0
min_j = 0
diff = 0
for i_offset in range(-movement, movement):
	for j_offset in range(-movement, movement):
		diff = 0
		for i in range(w_c - crop,w_c + crop):
			for j in range(h_c - crop,h_c + crop):
				if((0<=i+i_offset)&(i+i_offset < w)&(0<=j+j_offset)&(j+j_offset< h)):
					diff += (im_g[j, i]-mid) * (im_b[j+j_offset, i+i_offset]-mid)
	
		if(diff > min_d):
			min_d = diff
			min_i = i_offset
			min_j = j_offset

		#print("%3d %3d %d"%(i_offset, j_offset, diff))

print("Min: %3d %3d %d"%(min_i, min_j, min_d))

offib = min_i
offjb = min_j



min_d = 0
min_i = 0
min_j = 0
for i_offset in range(-movement, movement):
	for j_offset in range(-movement, movement):
		diff = 0
		for i in range(w_c - crop,w_c + crop):
			for j in range(h_c - crop,h_c + crop):
				if((0<=i+i_offset)&(i+i_offset < w)&(0<=j+j_offset)&(j+j_offset< h)):
					diff += (im_g[j, i]-mid) * (im_r[j+j_offset, i+i_offset]-mid)
					
		if(diff > min_d):
			min_d = diff
			min_i = i_offset
			min_j = j_offset

		#print("%3d %3d %d"%(i_offset, j_offset, diff))

print("Min: %3d %3d %d"%(min_i, min_j, min_d))

offir = min_i
offjr = min_j



rec_img = np.zeros((h,w,3), np.uint8)

rec_img[:,:,0] = im_g

for i in range(0,w):
	for j in range(0,h):
		if((0<=i+offib)&(i+offib < w)&(0<=j+offjb)&(j+offjb< h)):
			rec_img[j,i,1] = im_b[j+offjb, i+offib]


for i in range(0,w):
	for j in range(0,h):
		if((0<=i+offir)&(i+offir < w)&(0<=j+offjr)&(j+offjr< h)):
			rec_img[j,i,2] = im_r[j+offjr, i+offir]


cv2.imshow("cropped", rec_img)
cv2.waitKey(0)






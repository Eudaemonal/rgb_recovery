#!/usr/bin/python3.5

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt



# Match im2 onto im1
def match_offset(im1, im2, crop, movement):
	h,w = im1.shape
	w_c = int(w/2)
	h_c = int(h/2)

	# Find mid point of grey scale
	sample = im1[h_c - crop :h_c + crop, w_c - crop:w_c + crop]
	ret, imgf = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	mid = int(ret)

	max_diff = 0
	max_i = 0
	max_j = 0
	for i_offset in range(-movement, movement):
		for j_offset in range(-movement, movement):
			diff = 0
			for i in range(w_c - crop,w_c + crop):
				for j in range(h_c - crop,h_c + crop):
					if((0<=i+i_offset)&(i+i_offset < w)&(0<=j+j_offset)&(j+j_offset< h)):
						diff += (im1[j, i]-mid) * (im2[j+j_offset, i+i_offset]-mid)
					
			if(diff > max_diff):
				max_diff = diff
				max_i = i_offset
				max_j = j_offset

	print("max: %3d %3d %d"%(max_i, max_j, max_diff))

	return max_i, max_j 


def img_reconstruct(B, G, R, offjg, offig, offjr, offir):
	h, w = B.shape
	rec_img = np.zeros((h,w,3), np.uint8)

	# Reconstruct RGB channels
	rec_img[:,:,0] = B
	for i in range(0,w):
		for j in range(0,h):
			if((0<=i+offig)&(i+offig < w)&(0<=j+offjg)&(j+offjg< h)):
				rec_img[j,i,1] = G[j+offjg, i+offig]

	for i in range(0,w):
		for j in range(0,h):
			if((0<=i+offir)&(i+offir < w)&(0<=j+offjr)&(j+offjr< h)):
				rec_img[j,i,2] = R[j+offjr, i+offir]

	return rec_img



if __name__ == "__main__":
	img = cv2.imread('./DataSamples/s1.jpg',0)

	# Variable init
	height, width = img.shape 
	h = int(height/3)
	w = width
	movement = 20
	crop = 20

	# Create array of RGB values
	im_b = img[0 : h, 0:w] 
	im_g = img[h :2*h, 0:w]
	im_r = img[2*h : 3*h, 0:w]

	# Calculate offset using cross correlation
	offig, offjg = match_offset(im_b, im_g, crop, movement)
	offir, offjr = match_offset(im_b, im_r, crop, movement)

	# Create reconstruct image
	rec_img = img_reconstruct(im_b, im_g, im_r, offjg, offig, offjr, offir)

	cv2.imshow("cropped", rec_img)
	cv2.waitKey(0)



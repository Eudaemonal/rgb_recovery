#!/usr/bin/python3.5


import numpy as np
import cv2
import math


# Match im2 onto im1
def match_offset(im1, im2, movement):
	h, w = im1.shape
	# Find mid point of grey scale
	ret, imgf = cv2.threshold(im1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	mid = int(ret)

	max_diff = 0
	max_i = 0
	max_j = 0
	for i_offset in range(-movement, movement):
		for j_offset in range(-movement, movement):
			diff = 0
			for i in range(movement,w-movement):
				for j in range(movement,h-movement):
					if((0<=i+i_offset)&(i+i_offset < w)&(0<=j+j_offset)&(j+j_offset< h)):
						diff += (im1[j, i]-mid) * (im2[j+j_offset, i+i_offset]-mid)
					
			if(diff > max_diff):
				max_diff = diff
				max_i = i_offset
				max_j = j_offset

	print("max: %3d %3d %d"%(max_i, max_j, max_diff))

	return max_i, max_j 


def img_reconstruct(G, B, R, offjb, offib, offjr, offir):
	h, w = G.shape
	rec_img = np.zeros((h,w,3), np.uint8)

	# Reconstruct RGB channels
	rec_img[:,:,0] = G
	for i in range(0,w):
		for j in range(0,h):
			if((0<=i+offib)&(i+offib < w)&(0<=j+offjb)&(j+offjb< h)):
				rec_img[j,i,1] = B[j+offjb, i+offib]

	for i in range(0,w):
		for j in range(0,h):
			if((0<=i+offir)&(i+offir < w)&(0<=j+offjr)&(j+offjr< h)):
				rec_img[j,i,2] = R[j+offjr, i+offir]

	return rec_img

def img_pyramid(img, min_reso):
	h,w = img.shape
	step = int(math.log(w/min_reso)/math.log(2))
	img_arr = []
	img_arr.append(img)
	for i in range(0,step):
		img_arr.append(cv2.pyrDown(img_arr[i]))
	return img_arr



if __name__ == "__main__":
	img = cv2.imread('./DataSamples/s1.jpg',0)

	# Variable init
	height, width = img.shape 
	h = int(height/3)
	w = width
	movement = 5

	# Create array of RGB values
	im_g = img[0 : h, 0:w] 
	im_b = img[h : 2*h, 0:w]
	im_r = img[2*h : 3*h, 0:w]

	min_reso = 8
	# Construct image pyramid
	arr_g = img_pyramid(im_g, min_reso)
	arr_b = img_pyramid(im_b, min_reso)
	arr_r = img_pyramid(im_r, min_reso)

	idx = 2
	# Calculate offset using cross correlation
	offib, offjb = match_offset(arr_g[idx], arr_b[idx], movement)
	offir, offjr = match_offset(arr_g[idx], arr_r[idx], movement)

	offib = offib* 2**idx
	offjb = offjb* 2**idx
	offir = offir* 2**idx
	offjr = offjr* 2**idx

	# Create reconstruct image
	rec_img = img_reconstruct(im_g, im_b, im_r, offjb, offib, offjr, offir)

	cv2.imshow("cropped", rec_img)
	cv2.waitKey(0)
	#lower_reso = cv2.pyrDown(im_g)

	#cv2.imshow("high", im_g)
	#cv2.imshow("low", lower_reso)
	#cv2.waitKey(0)

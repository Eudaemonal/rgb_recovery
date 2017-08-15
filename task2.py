#!/usr/bin/python3.5


import numpy as np
import cv2
import math


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
					if((0<=i+i_offset)&(i+i_offset < w)&(0<=j+j_offset)&(j+j_offset< h)&(0<=i)&(i< w)&(0<=j)&(j< h)):
						diff += (im1[j, i]-mid) * (im2[j+j_offset, i+i_offset]-mid)
					
			if(diff > max_diff):
				max_diff = diff
				max_i = i_offset
				max_j = j_offset

	return max_i, max_j 

# Move image based on offset
def move_image(img, offj, offi):
	h, w = img.shape
	ret_img = np.zeros((h,w,1), np.uint8)
	for i in range(0,w):
		for j in range(0,h):
			if((0<=i+offi)&(i+offi < w)&(0<=j+offj)&(j+offj< h)):
				ret_img[j,i] = img[j+offj, i+offi]
	return ret_img


# Reconstruct RGB image
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

# Construct image pyramid
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
	movement = 2
	crop = 120
	min_reso = int(w/10)


	# Create array of RGB values
	im_b = img[0 : h, 0:w] 
	im_g = img[h : 2*h, 0:w]
	im_r = img[2*h : 3*h, 0:w]

	# Construct image pyramid
	arr_b = img_pyramid(im_b, min_reso)
	arr_g = img_pyramid(im_g, min_reso)
	arr_r = img_pyramid(im_r, min_reso)

	idx = len(arr_g) -1
	
	while(idx>0):
		print("idx: %d"%(idx))
		# Calculate offset using cross correlation
		offig, offjg = match_offset(arr_b[idx], arr_g[idx],movement, movement)
		offir, offjr = match_offset(arr_b[idx], arr_r[idx],movement, movement)

		print("max: %3d %3d"%(offig*2**idx, offjg*2**idx))
		print("max: %3d %3d"%(offir*2**idx, offjr*2**idx))


		move_image(arr_b[idx-1], offjg*2, offig*2)
		move_image(arr_r[idx-1], offjr*2, offir*2)

		idx = idx - 1

	offig, offjg = match_offset(arr_b[idx], arr_g[idx],crop, movement)
	offir, offjr = match_offset(arr_b[idx], arr_r[idx],crop, movement)

	print("idx: %d"%(idx))
	print("max: %3d %3d"%(offig*2**idx, offjg*2**idx))
	print("max: %3d %3d"%(offir*2**idx, offjr*2**idx))

	# Create reconstruct image
	rec_img = img_reconstruct(arr_b[0], arr_g[0], arr_r[0], offjg, offig, offjr, offir)

	cv2.imshow("cropped", rec_img)
	cv2.waitKey(0)

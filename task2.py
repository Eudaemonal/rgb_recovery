#!/usr/bin/python3.5


import numpy as np
import cv2
import math


# Match im2 onto im1
def match_offset(im1, im2, crop, movement):
	h,w = im1.shape
	w_c = int(w/2)
	h_c = int(h/2)
	if(crop>100):
		crop = 100

	w_window = int(w_c*crop/100)
	h_window = int(h_c*crop/100)

	# Find mid point of grey scale
	sample = im1[h_c - h_window :h_c + h_window, w_c - w_window:w_c + w_window]
	ret, imgf = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	mid = int(ret)

	max_diff = 0
	max_i = 0
	max_j = 0
	for i_offset in range(-movement, movement):
		for j_offset in range(-movement, movement):
			diff = 0
			for i in range(w_c - w_window,w_c + w_window):
				for j in range(h_c - h_window,h_c + h_window):
					if((0<=i)&(i < w)&(0<=j)&(j< h)&(0<=i+i_offset)&(i+i_offset < w)&(0<=j+j_offset)&(j+j_offset< h)):
						diff += (im1[j, i]-mid) * (im2[j+j_offset, i+i_offset]-mid)
					
			if(diff > max_diff):
				max_diff = diff
				max_i = i_offset
				max_j = j_offset


	return max_i, max_j 


# Move image based on offset
def move_image(img, offj, offi):
	h, w = img.shape
	ret_img = np.zeros((h,w), np.uint8)
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


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged


if __name__ == "__main__":
	img = cv2.imread('./DataSamples/s1.jpg',0)

	# Variable init
	height, width = img.shape 
	h = int(height/3)
	w = width
	movement = 2
	crop = 20
	min_reso = 32 # int(w/10)

	# Create array of RGB values
	im_b = img[0 : h, 0:w] 
	im_g = img[h : 2*h, 0:w]
	im_r = img[2*h : 3*h, 0:w]

	# Using edges for alignment
	ime_b = auto_canny(im_b)
	ime_g = auto_canny(im_g)
	ime_r = auto_canny(im_r)

	# Construct image pyramid
	arr_b = img_pyramid(ime_b, min_reso)
	arr_g = img_pyramid(ime_g, min_reso)
	arr_r = img_pyramid(ime_r, min_reso)

	idx = len(arr_g) -1
	
	ig = 0
	jg = 0
	ir = 0
	jr = 0

	while(idx>0):
		print("idx: %d"%(idx))
		# Calculate offset using cross correlation
		offig, offjg = match_offset(arr_b[idx], arr_g[idx],100, movement)
		offir, offjr = match_offset(arr_b[idx], arr_r[idx],100, movement)

		print("max: %3d %3d"%(offig*2**idx, offjg*2**idx))
		print("max: %3d %3d"%(offir*2**idx, offjr*2**idx))

		ig = ig*2 + offig*2
		jg = jg*2 + offjg*2
		ir = ir*2 + offir*2
		jr = jr*2 + offjr*2

		arr_g[idx-1] = move_image(arr_g[idx-1], jg, ig)
		arr_r[idx-1] = move_image(arr_r[idx-1], jr, ir)

		idx = idx - 1

	offig, offjg = match_offset(arr_b[idx], arr_g[idx],crop, movement)
	offir, offjr = match_offset(arr_b[idx], arr_r[idx],crop, movement)

	print("idx: %d"%(idx))
	print("max: %3d %3d"%(offig, offjg))
	print("max: %3d %3d"%(offir, offjr))
	ig = ig + offig
	jg = jg + offjg
	ir = ir + offir
	jr = jr + offjr

	print("final max: %3d %3d"%(ig, jg))
	print("final max: %3d %3d"%(ir, jr))

	# Create reconstruct image
	rec_img = img_reconstruct(im_b, im_g, im_r, jg, ig, jr, ir)

	cv2.imshow("reconstructed", rec_img)
	cv2.waitKey(0)

#!/usr/bin/env python3
import cv2
import pygame
import numpy as np
from matplotlib import pyplot as plt

W = 1920//2
H = 1080//2

#orb feature detection
orb = cv2.ORB_create()

pdes = np.array([])

def feature_tracking(img, pdes):
	kp = orb.detect(img, None)
	kp, des = orb.compute(img, kp)
	#print(pdes.shape)
	match_des(pdes, des)
	img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=1)
	#img2 = cv2.drawKeypoints(img, kp, np.array([]), (0, 0, 255), flags=4)
	plt.show()
	pdes = des
	return img2, pdes
 
def match_des(pdes, des):
	match = 0
	if len(pdes) != 0:
		for i in np.arange(len(pdes)):
			if pdes[i].all() == des[i].all():
				match += 1
		print("Matches:", match)
	else:
		print("None")

def process_frame(imgarray):
	cv2.imwrite('color.jpg', imgarray)
	cv2.imshow('image', imgarray)
	cv2.waitKey(1)
	#print(imgarray.shape)

cap = cv2.VideoCapture('slam.mp4')	

while(cap.isOpened()):
	ret, frame =  cap.read()
	if ret == True:	
		img, pdes = feature_tracking(frame, pdes)
		process_frame(img)
	else:
		break
	

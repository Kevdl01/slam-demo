#!/usr/bin/env python3
import cv2
import pygame

W = 1920//2
H = 1080//2

def process_frame(imgarray):
	cv2.imwrite('color.jpg', imgarray)
	cv2.imshow('image', imgarray)
	cv2.waitKey(1)
	print(imgarray.shape)

cap = cv2.VideoCapture('slam.mp4')	

while(cap.isOpened()):
	ret, frame =  cap.read()
	if ret == True:
		process_frame(frame)
	else:
		break
	

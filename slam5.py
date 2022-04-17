#!/usr/bin/env python3
import cv2
import numpy as np
from matplotlib import pyplot as plt

pdes = np.array([])
kp1 = ()
img1 = np.array([])
							#fundamental matrix filter matches
class detect(object):
	def __init__(self):
		self.orb = cv2.ORB_create()                                   #ORB feature detector
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)    #bruteforce matcher
		self.sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=30, contrastThreshold=0.9)
		self.bfk = cv2.BFMatcher()

	def keypoints(self, img, pdes, kp1, img1):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		corners = cv2.goodFeaturesToTrack(gray,2000,0.01,3)
		kps = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in corners]
		kps, des = self.orb.compute(img, kps)
		#kps, des = self.sift.compute(img, kps)
		corners = np.int0(corners)
		matches = self.frame_match(pdes, des)                        #list of objects with attribs - queryIdx, trainIdx
		if len(pdes)>0:
			matches = ((kp1[m.queryIdx].pt, kps[m.trainIdx].pt) for m in matches)  
			self.fmfilter(img1, img, matches)
			#self.showlines(img, matches)	
		kp1 = kps 
		img1 = np.copy(img)
		pdes = des
		return corners, pdes, kp1, img1
	
	def showlines(self, img2, matches):
		for pt1, pt2 in matches:
			u1, v1 = map(lambda x: int(round(x)), pt1)
			u2, v2 = map(lambda x: int(round(x)), pt2)
			img3 = cv2.line(img2, (u1, v1), (u2, v2), (0, 255, 0), 2)
		cv2.imshow('image', img3)
		cv2.waitKey(1)
	
	def frame_match(self, pdes, des):
		good_match = []
		if len(pdes) != 0:	
			#matches = self.bf.match(pdes, des)  
			#matches = self.bfk.knnMatch(pdes, des, k=2)
			matches = self.bfk.knnMatch(pdes, des, k=1)
			for m in matches:
				if m[0].distance < 79.75:
					good_match.append(m[0])
			#for m,n in matches:
			#	if m.distance < 0.55*n.distance:
			#		good_match.append(m)
			#print(m.distance for m in matches)
			#print(matches[0].distance)
			matches = sorted(good_match, key=lambda x:x.distance)
			return matches
		else:
			print("None")

	#filter FM
	def fmfilter(self, img1, img2, matches):
		p1 = []
		p2 = []
		for pt1, pt2 in matches:
			p1.append(pt1)
			p2.append(pt2)
		p1 = np.int32(p1)
		p2 = np.int32(p2)
		F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_LMEDS)
		if mask is not None:
			p1 = p1[mask.ravel()==1]
			p2 = p2[mask.ravel()==1]
		# 	print(p1.shape, p2.shape)
			l1 = cv2.computeCorrespondEpilines(p2.reshape(-1,1,2), 2,F)
			l1 = l1.reshape(-1, 3)
			img5,img6 = self.drawlines(img1,img2,l1,p1,p2)
			#print(img5)
			cv2.imshow('image', img5)
			cv2.waitKey(1)

	def drawlines(self, img1, img2, lines, pts1, pts2):
		#print(img1.shape)
		r, c, j = img1.shape
		#img1 = cv.cvtColor(img1,cv2.COLOR_GRAY2BGR)
		#img2 = cv.cvtColor(img2,cv2.COLOR_GRAY2BGR)
		for r,pt1,pt2 in zip(lines,pts1,pts2):
			color = tuple(np.random.randint(0,255,3).tolist())
			x0,y0 = map(int, [0, -r[2]/r[1] ])
			x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
			img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
			img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
			img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
		return img1,img2

def display(imgarray, corners, mask=0):
	cv2.imwrite('color.jpg', imgarray)
	if mask == 0:
		for i in corners:
			x, y = i.ravel()
			cv2.circle(imgarray, (x,y), radius=3, color=(0,255,0))
		cv2.imshow('image', imgarray)
	else:
		cv2.imshow('masked', mask)
	cv2.waitKey(1)
	print(imgarray.shape)

d = detect()
cap = cv2.VideoCapture('slam.mp4')	

while(cap.isOpened()):
	r, frame =  cap.read()
	if r == True:	
		corners, pdes, kp1, img1 = d.keypoints(frame, pdes, kp1, img1)
		#display(frame, corners)
	else:
		break
	

# Extracting features using optical flow
# walking:1, Jogging:2, Handclapping: 3, Handwaving: 4, running: 5, boxing: 6 	
#---------------------------------------------------------------------------------

import numpy as np
import cv2
import imutils
from sklearn.cluster import KMeans
import math
import os

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
	                   qualityLevel = 0.3,
	                   minDistance = 7,
	                   blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
	              maxLevel = 2,
	              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def ExtractFeatures(cap):
	# Create some random colors
	color = np.random.randint(0,255,(100,3))

	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	# old_frame = imutils.resize(old_frame, width=128, height=128)
	old_frame = cv2.resize(old_frame, (128,128), interpolation = cv2.INTER_AREA)
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	
	# w, h = len(old_frame), len(old_frame[0])
	Frames = np.empty([0, 128*128], dtype="uint8")  # Frames -> will store the features in which each row will represent the one frame of the video. 

	# Create a maxCornersk image for drawing purposes
	mask = np.zeros_like(old_frame)
	# print("mask.shape",mask.shape)
	count = 0
	while True:
	    ret,frame = cap.read()
	    count+=1
	    if not ret:
	    	break

	    frame = cv2.resize(frame, (128,128), interpolation = cv2.INTER_AREA)
	    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    # calculate optical flow
	    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray , p0, None, **lk_params)

	    # Select good points
	    if np.any(st==1):	
			good_new = p1[st==1]
			good_old = p0[st==1]

	    # draw the tracks
	    for i, (new, old) in enumerate(zip(good_new, good_old)):
	    	a,b = new.ravel()
	    	c,d = old.ravel()
	    	mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
	    	frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
	    
	    img = cv2.add(frame,mask)
	    # print("*", mask.shape)

	    img = imutils.resize(img, width=500)
	    # mask = imutils.resize(mask, width=500)
	    
	    # print ("len(mask):", len(mask))
	    m = np.around(mask[:,:,1].ravel(), decimals=3)

	    Frames = np.vstack((Frames, m))

	    cv2.imshow('frame',img)
	    cv2.imshow('mask', mask[:,:,1])
	    k = cv2.waitKey(30) & 0xff
	    if k == 27:
	        break
	    
	    # Now update the previous frame and previous points
	    old_gray = frame_gray.copy()
	    p0 = good_new.reshape(-1, 1, 2)

	print ("count:", count)
	np.savetxt("Feature.txt", Frames)

	cv2.destroyAllWindows()
	cap.release()
	return count

def main():
	# labels = list()
	global labels
	labels = {'walking':1, 'jogging':2, 'handclapping': 3, 'handwaving': 4, 'running': 5, 'boxing': 6 }
	i=0
	j=0
	filename = []
	Feature_labels = list()
	directory = os.path.join("c:\\","/home/chandu/Activity Anticipation/videos")
	for root,dirs,files in os.walk(directory):
		for file in files:
			filename.append(file)

	for it in range(len(filename)):
		temp = list()
		cap = cv2.VideoCapture("videos/"+filename[it])
		count = ExtractFeatures(cap)
		print ("###", type(labels))
		name = filename[it].split('.')
		temp.append(int(labels[name[0]]))
		temp = temp*count
		Feature_labels.append(temp)
	print ("Feature_labels: ", Feature_labels)


if __name__=="__main__":
	main()
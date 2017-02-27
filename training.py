import numpy as np
import cv2
import imutils
from sklearn.cluster import KMeans
import math

cap = cv2.VideoCapture("videos/person15_running_d1_uncomp.avi")


	# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
	                   qualityLevel = 0.3,
	                   minDistance = 7,
	                   blockSize = 7 )

	# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
	              maxLevel = 2,
	              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def ExtractFeatures():
	# Create some random colors
	color = np.random.randint(0,255,(100,3))

	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	w, h = len(old_frame), len(old_frame[0])
	Frames = np.empty([0, w*h], dtype="uint8")


	# Create a maxCornersk image for drawing purposes
	mask = np.zeros_like(old_frame)
	print("mask.shape",mask.shape)
	count = 0
	while True:
	    ret,frame = cap.read()
	    count+=1
	    
	    if not ret:
	    	break

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
	    
	    m = np.around(mask[:,:,1].ravel(), decimals=3)

	    Frames = np.vstack((Frames, m))

	    cv2.imshow('frame',img)
	    cv2.imshow('mask', mask[:,:,1])
	    k = cv2.waitKey(30) & 0xff
	    if k == 27:
	        breakr
	    
	    # Now update the previous frame and previous points
	    old_gray = frame_gray.copy()
	    p0 = good_new.reshape(-1, 1, 2)

	print ("count:", count)
	np.savetxt("Feature.txt", Frames)

	cv2.destroyAllWindows()
	cap.release()
	return Frames


def Clustering(Frames):
	
	iter_=2
	prev_var=0
	while iter_ < len(Frames)/2:
		kmeans = KMeans(n_clusters=iter_, random_state=0).fit(Frames)
		new_var = 0
		for j in range(iter_):
			indices = [i for i, x in enumerate(kmeans.labels_) if x == j]
			li=list()
			for k in range(len(indices)):
				li.append(Frames[k])
			li = np.asarray(li)
			new_var+=np.var(li)
		
		new_var = math.sqrt(new_var)
		print(iter_)
		print("new_var:", new_var)

		if iter_==2:
			prev_var = new_var
		elif new_var < prev_var:
			prev_var = new_var
		else:
			break
			
		iter_+=1
		prev_kmeansLabels = kmeans.labels_

	return iter_, prev_kmeansLabels

def main():
	Feature = ExtractFeatures()
	n, labels = Clustering(Feature)
	print("no. of lables:", n)
	print("*", labels)


if __name__=="__main__":
	main()
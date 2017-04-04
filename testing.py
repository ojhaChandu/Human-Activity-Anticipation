import cv2
import numpy as np
from ExtractingFeature import ExtractFeatures
from DenseOptFeatureExtractor import DenseOptFlow
import cPickle
import os
def main():

	feature_mat = np.empty([0, 128*128], dtype="uint8")  # Frames -> will store the features in which each row will represent the one frame of the video. 
	filename = list()
	directory = os.path.join("c:\\","/home/chandu/Activity Anticipation/test")
	for root,dirs,files in os.walk(directory):
		for file in files:
			filename.append(file)
	print(filename)
	for i in range(len(filename)):
		cap = cv2.VideoCapture("test/"+filename[i])
		# (count, feature_mat) = ExtractFeatures(cap, feature_mat)
		(count, feature_mat) = DenseOptFlow(cap, feature_mat)
		
	data = cPickle.load(open("Svm_classifier.pkl", "rb"))
	print(repr(data))
	labels = data.predict(feature_mat)
	print ("labels", labels.shape)
	print(np.histogram(labels,6))

if __name__=="__main__":
	main()
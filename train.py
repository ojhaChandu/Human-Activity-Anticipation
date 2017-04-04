import numpy as np
from numpy import *
import cPickle


def SvmTrain():
	Feature_labels = loadtxt("Dense_Feature_labels.txt")
	Features = loadtxt("Dense_Feature.txt")
	# print(Features.shape)
	# print(Feature_labels.shape)
	from sklearn import svm	
	clf = svm.SVC()
	clf.fit(Features, Feature_labels)
	S = open('Svm_classifier.pkl', 'wb')
	cPickle.dump(clf, S, -1)

def main():
	# CRFtraining()
	SvmTrain()
if __name__=="__main__":
	main()
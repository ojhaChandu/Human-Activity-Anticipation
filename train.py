import sklearn_crfsuite
import numpy as np
from numpy import *
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


def training():
	crf = sklearn_crfsuite.CRF(
	    algorithm='lbfgs',
	    c1=0.1,
	    c2=0.1,
	    max_iterations=100,
	    all_possible_transitions=True)
	Feature_labels = loadtxt("Feature_labels.txt")
	Features = loadtxt("Feature.txt")
	print(Features[1].shape)
	if np.any(Features > 0):
		print True
	# Feature_labels = np.asarray(Feature_labels)[np.newaxis]
	# Feature_labels = Feature_labels.T
	# print (Feature_labels.shape)
	# Feature_labels = Feature_labels.astype(np.int)
	#Feature_labels = list(Feature_labels)
	#Features = list(Features)

	# Features = np.asarray(Features)
	l = len(Features)
	print (type(Features[1]))
	# print (Feature_labels)
	# Feature_labels = [int(x) for x in Feature_labels]
	# Features = [int(y) for y in x for x in Features]
	#for i in range(l):
	#		Features[i] = list(Features[i])

	# print (Features[1])
	crf.fit(Features, Feature_labels)

def main():
	training()

if __name__=="__main__":
	main()
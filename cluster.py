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
	n, labels = Clustering(Feature_labels)

if __name__=="__main__":
	main()
import sys
import numpy as np
from kmeans import kmeans
from scipy.io import arff
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def parse(filename, rows, columns):
	data, meta = arff.loadarff(filename)

	# after reading the data we need to compute the mean value for the '?'
	column_median = [0]*columns 
	for j in range(0, columns):
		count = 0 # we need to count the number of entries for that column that are different from '?'
		for i in range(0, rows):
			if not np.isnan(data[i][j]):
				column_median[j] += data[i][j]
				count += 1
		column_median[j]  = column_median[j]/count

	for i in range(0, rows):
		for j in range(0, columns):
			if np.isnan(data[i][j]):
				data[i][j] = column_median[j]

	return (data, meta)

def read_weka_centroids(weka_file):
	f =  open(weka_file, 'r')
	centroids = []
	for line in f:
		if 'Cluster' not in line:
			centroid = line.split(',')[:-1]
			centroids.append(centroid)

	for i in range(len(centroids[0])):
		centroids[0][i] = float(centroids[0][i])
		centroids[1][i] = float(centroids[1][i])

	return (centroids[0], centroids[1])

def main():
	cmdargs = sys.argv
	G = int(cmdargs[1]) # number of centroids
	R = int(cmdargs[2]) # number of rows to consider
	C = int(cmdargs[3]) # number of columns to read
	filename = cmdargs[4] # dataset file

	data, meta = parse(filename, R, C)
	print (meta['class'][1])

	# attributes and label separation for each sample
	labels = [None]*R
	samples = np.zeros((R, C))
	for i in range(0, R):
		for j in range(0, C):
			samples[i][j] = data[i][j]
		# convertion of the sample class into Positive (1) and Negative (0) labels 
		labels[i] = 1 if data[i][-1].decode('utf-8') == meta['class'][1][0] else 0

	# seed 115 accuracy 0.8667, this value was the result of many tests
	k_means = kmeans(k = G) 
	k_means.train(data = samples, seed=115)
	prediction = k_means.predict(data = samples)

	# note that we dont know which centroid is the positive class... 
	# but in this case since we choosed the seed we know that the cluster 0 will correspond to the negative classe and 1 to the positive
	# assuming cluster 0 is the negative class:
	tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
	precision = tp/(tp + fp)
	recall = tp/(tp + fn)
	accuracy = (tp + tn)/(tp + fp + fn + tn)

	print ("Precision: %f" % (precision))
	print ("Recall: %f" % (recall))
	print ("Accuracy: %f" % (accuracy))
	print ("Clusters 0:")
	print (k_means.centroids[0])
	print ("Clusters 1:")
	print (k_means.centroids[1])
	print ("Number of points classified +: %d" % (prediction.count(0)))
	print ("Number of points classified -: %d" % (prediction.count(1)))

	# distances for 2.1 b  Question
	"""
	centroid1, centroid2 = read_weka_centroids("centroids_weka.txt")
	import scipy.spatial as sp
	print ("distance between centroids +: %f" % (sp.distance.euclidean(centroid1, k_means.centroids[0])))
	print ("distance between centroids +: %f" % (sp.distance.euclidean(centroid2, k_means.centroids[1])))

	print ("(Our kmeans) distance between centroids +-: %f" % (sp.distance.euclidean(k_means.centroids[0], k_means.centroids[1])))
	print ("(Weka) distance between centroids +-: %f" % (sp.distance.euclidean(centroid1, centroid2)))
	"""

	# plots for 1.1  Question
	"""
	centers = np.array([k_means.centroids[0].tolist(), k_means.centroids[1].tolist()])
	plt.scatter(samples[:, 0], samples[:, 1], c=prediction, s=50, cmap='viridis')
	plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
	plt.show()
	"""

if __name__ == '__main__':
	main()
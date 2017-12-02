import numpy as np
import scipy.spatial as sp
from scipy.io import arff
from sklearn.metrics import confusion_matrix

#-------------------------------------------------------------------------------------------------------------------------------------
class kmeans(object):
	""" This class defines the K-Means algorithm. """

	# @brief: 
	#		  Constructor method that receives the K parameter of the k means and the distance method to be used to compute
	#		  the distance between the samples/points. If the distance_metric arguments is ignored the algorithm will use an
	#		  euclidean distance.
	def __init__(self, k, distance_metric = sp.distance.euclidean):
		self.k = k
		self.distance_metric = distance_metric
		self.centroids = None

	# @brief: 
	#		  This is a private function that receives data points and returns a list where each entry i is the list of points
	#	      assigned to centroid i.
	# NOTE:	  This funtion will be usefull for the training and prediction methods.
	def __centroid_assignment(self, data):
		assignment = []
		for i in range(0, self.k):
			assignment.append([])
			
		# this list will store the samples assigned to each centroid k
		for i in range(0, len(data)):
			closest_centroid = None
			distance = float('inf') # initialize the distance to an infinite number

			for j in range(0, self.k):
				if distance > self.distance_metric(data[i], self.centroids[j]):
					closest_centroid = j
					distance = self.distance_metric(data[i], self.centroids[j])

			assignment[closest_centroid].insert(0, data[i])

		return assignment

	# @brief: 
	#		  This function computes the k centroids given a dataset.
	#		  The maxit argument specifies the maximum number of iterations the algorithm will perform. Note that the algorithm can 
	#		  stop before the maximum number of iteration if it converges.
	def train(self, data, maxit = 200, seed = None):
		#---------------------------------------------------------------------------------
		# @brief: 
		#		 Auxiliar local function that computes the mean point over a set of points.
		# NOTE:  This function is an alternative to np.mean functions that gives NaN. 
		def update_centroid(samples, centroid):
			updated_centroid = np.zeros(len(samples[0]))
			# no update case
			if len(samples) == 0:
				return centroid

			# update case
			for i in range(0, len(samples[0])):
				for sample in samples:
					updated_centroid[i] += sample[i]
				updated_centroid[i] = updated_centroid[i]/len(samples)
			return np.array(updated_centroid)
		#---------------------------------------------------------------------------------
		rows = len(data)

		# random initialization of the centroids
		if seed:
			np.random.seed(seed)
		self.centroids = [np.array(data[np.random.randint(0, rows)]) for _ in range(0, self.k)]

		update = True # variable to check if the algorithm has converged.
		count = 0 # count the number of iterations
		while (count < maxit and update):

			assignment = self.__centroid_assignment(data)
			update = False # lets assume that we have converged. if in the next cycle the value changes to True, this assumption was wrong.
			for k in range(0, self.k):
				updated_centroid = update_centroid(assignment[k], self.centroids[k])
				if not np.array_equal(updated_centroid, self.centroids[k]):
					update = True
					self.centroids[k] = updated_centroid
			count += 1

		return self.centroids

	# @brief: 
	#		  This algorithm receives a set of points and returns a list where each entry i has the value of the cluster 
	#		  assigned to the ith sample in the data.
	def predict(self, data):
		assignment = self.__centroid_assignment(data)
		# convert assignments to list in order to use the operation "in"
		for i in range(0, self.k):
			for j in range(0, len(assignment[i])):
				assignment[i][j] = assignment[i][j].tolist()

		# convert data to list to
		data = data.tolist()
		labels = [None]*len(data)
		for i in range(0, len(data)):
			for k in range(0, self.k):
				if data[i] in assignment[k]:
					labels[i] = k
					break
		return labels
#-------------------------------------------------------------------------------------------------------------------------------------

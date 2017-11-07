import numpy as np 
import matplotlib.pyplot as plt
class KMean():
	def __init__(self, n_clusters):
		self.n_clusters = n_clusters
	#this returns the Euclidean distancec squared
	def Euclidean_distance(self, point1, point2):
		return sum([(point1[i] -point2[i])**2 for i in range(len(point1))])
	#this method find the centroids of the data set. Detailed steps below:
	#- n_cluster data points are chosen randomly among the given data
	#- each point in the data set is assigned to its closest centroid.
	#- the new centroid set is calculated based on the average of the data points corresponding to 
	#each old centroids
	#- the last two steps are repeated many time for the centroids to get to theirs final position.
	def fit(self, data):
		self.n_epoches = 20
		matrix = np.array(data)
		self.n_samples= len(matrix)
		self.n_features = len(data[0])
		self.centroids = [np.random.randint(0, self.n_samples) for i in range(self.n_clusters)]
		self.centroids = np.array([matrix[i] for i in self.centroids])
		self.trace = []
		for epoch in range(self.n_epoches):
			self.clusters = [[] for i in range(self.n_clusters)]

			for i in range(self.n_samples):
				self.clusters[np.argmin([self.Euclidean_distance(matrix[i], self.centroids[j]) for j in range(self.n_clusters)])].append(i)

			for i in range(self.n_clusters):
				self.centroids[i, :] = np.mean(matrix[self.clusters[i]], axis = 0)
	#This method predict the nearest centroids for a given data set.
	def predict(self, data):
		matrix = np.array(data)
		if len(matrix.shape) == 1:
			try:
				if len(matrix) != self.n_features:
					raise NameError('Prediction input size does not match training data')
				else:
					return np.argmin([self.Euclidean_distance(matrix, self.centroids[j]) for j in range(self.n_clusters)])
			except NameError:
				print('Error occur')
		else:
			try:
				if matrix.shape[1] != self.n_features:
					raise NameError('Prediction input size does not match training data')
				else:
					return np.array([self.predict(matrix[i]) for i in range(len(matrix))])
			except NameError:
				print('Error occur')
				raise
if __name__=='__main__':
	kmean = KMean(5)
	data = np.genfromtxt('data.csv', delimiter = ', ')
	kmean.fit(data)
	xmin, xmax = -3, 13
	ymin, ymax = -3, 13
	h = 0.05
	xx, yy = np.meshgrid(np.arange(xmin, xmax,h), np.arange(ymin, ymax, h))
	Z = kmean.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
	           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	           cmap=plt.cm.Paired,
	           aspect='auto', origin='lower')
	plt.scatter(data[:, 0], data[:, 1], s = 5, c = 'black')
	plt.scatter(kmean.centroids[:, 0], kmean.centroids[:, 1 ], marker='o', s = 100, c = 'red')
	plt.title('K-Mean example')
	plt.show()

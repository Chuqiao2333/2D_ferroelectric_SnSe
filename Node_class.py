
import numpy           as     np
from   sklearn.cluster import KMeans
from   sklearn.cluster import MiniBatchKMeans
from   sklearn.cluster import SpectralClustering
import umap
import helper_function as hf

class ClusterNode(object):
	"""docstring for Cluster_Node"""
	def __init__(self, data, data_shape, ancestor = None, sub_idx = -1):
		super(ClusterNode, self).__init__()

		self.data  = data
		self.data_shape = data_shape
		self.ancestor   = ancestor
		self.sub_idx    = sub_idx
		self.center    = None

	def vectorize(self):
		if self.sub_idx == -1:
			x,y,kx, ky = self.data_shape

			self.data  = np.reshape(self.data, (x*y,kx*ky))
		else:
			print('Please run this method on the root node')

	def Alignment(self, threshold = 0.1):

		if self.sub_idx == -1:
			self.data, mean, std = hf.alignment(self.data)
			


			self.center = mean
		else:
			print('Please run this method on the root node')

	def visualization(self):

		if self.sub_idx == -1:
			x,y,kx, ky = self.data_shape
			embedding = []

			for i in range(len(self.data)):
			  
				idx = np.array([x - int(i%x), int(i//y)])
				embedding.append(idx)
			embedding = np.array(embedding)


			labels = np.reshape(np.round((self.ADF - np.min(self.ADF))/np.ptp(self.ADF) * 255)  ,(x*y,))

			hf.visualize2DMAP(self.data,embedding, np.int32(labels), 256, self.data_shape, "Wide")
		else:
			print('Please run this method on the root node')

	def addMasks(self, in_r, out_r):

		if self.sub_idx == -1:
			center_mask = np.ones((self.data_shape[2], self.data_shape[3]))

			if self.center == None:
				center_idx = [self.data_shape[2]/ 2 - 0.5, self.data_shape[3]/2 - 0.5]
			else:
				center_idx = self.center

			for i in range(self.data_shape[2]):
				for j in range(self.data_shape[3]):
					if (i - center_idx[0])**2 + (j - center_idx[1])**2 <  in_r ** 2:
						center_mask[i][j] = 0
					if (i - center_idx[0])**2 + (j - center_idx[1])**2 >  out_r** 2:
						center_mask[i][j] = 0

			self.data = self.data * np.reshape(center_mask, (1, self.data_shape[2] *  self.data_shape[3]))

			return center_mask
		else:
			print('Please run this method on the root node')
		
	def getMaxDiffraction(self):

		if self.sub_idx == -1:
			max_diff    = np.max(self.data,axis = 0)
			self.max_diff_pattern = np.reshape(max_diff,
								(self.data_shape[2],self.data_shape[3]))

			return self.max_diff_pattern
		else:
			print('Please run this method on the root node')

	def getMeanDiffraction(self):

		if self.sub_idx == -1:
			mean_diff    = np.mean(self.data,axis = 0)
			self.mean_diff_pattern = np.reshape(mean_diff,
								(self.data_shape[2],self.data_shape[3]))

			return self.mean_diff_pattern
		else:
			print('Please run this method on the root node')

	def getADF(self):

		if self.sub_idx == -1:

			self.ADF = hf.quickHAADF(self.data)

			return self.ADF
		else:
			print('Please run this method on the root node')

	def getMainifoldStructure(self, n_neighbors = 10, min_dist = 0.1, n_components = 3):

		fit = umap.UMAP(
				n_neighbors  = n_neighbors,
				min_dist     = min_dist,
				n_components = n_components,
				)
		self.mainifold_structure  = fit.fit_transform(self.data)

		return self.mainifold_structure

	def getSSELine(self, kmax):

		sse = []
		for k in range(1, kmax+1):
			kmeans = MiniBatchKMeans(n_clusters = k).fit(self.data)
			centroids = kmeans.cluster_centers_
			pred_clusters = kmeans.predict(self.data)
			curr_sse = 0
			
			# calculate square of Euclidean distance of each point from its cluster center and add to current WSS
			for i in range(len(self.data)):
				curr_center = centroids[pred_clusters[i]]
				curr_sse += np.mean((self.data[i] - curr_center) ** 2 )

	
			sse.append(curr_sse)
			print('k = {} is finished'.format(k))
		return sse
	def getClusterNumber(self):

		print('Input CLuster Numbers')
		self.k = int(input())

	def getClusterResults(self, method = 'KMeans', batch_size = 100):

		if method == 'KMeans':
			cluster_result = KMeans(n_clusters = self.k).fit(self.data)

		elif method == 'MiniBatchKMeans':
			cluster_result = MiniBatchKMeans(n_clusters = self.k, batch_size = batch_size).fit(self.data)

		elif method == 'SpectralClustering':
			norm_data = ( self.data - np.min( self.data)) / np.ptp( self.data)
			cluster_result = SpectralClustering(n_clusters = self.k, assign_labels="discretize").fit(norm_data)

		self.labels  = cluster_result.labels_
		if method == 'SpectralClustering':

			self.centers = hf.getClusterCenters(self.data, self.k, cluster_result.labels_)

		else:

			self.centers = cluster_result.cluster_centers_

		return self.labels, self.centers


	def getRealSpaceMap(self):

		curr_node = self
		curr_label = self.labels.copy()
		while curr_node.ancestor != None:
			temp_label = curr_node.ancestor.labels.copy()
			sel = temp_label != curr_node.sub_idx
			temp_label[sel] = -1
			sel = temp_label == curr_node.sub_idx
			temp_label[sel] = curr_label.copy()

			curr_node = curr_node.ancestor
			curr_label = temp_label.copy()
		
		return np.reshape(curr_label, (self.data_shape[0], self.data_shape[1]))


	def getSubClusters(self):

		sub_cluster_data_list = [ [] for i in range(self.k) ]
		for i in range(len(self.labels)):
			idx = self.labels[i]
			sub_cluster_data_list[idx].append(self.data[i])

		self.sub_clusters = []
		for i, sub_list in enumerate(sub_cluster_data_list):
			self.sub_clusters.append(ClusterNode(data = np.array(sub_list), data_shape = self.data_shape, ancestor = self, sub_idx = i))










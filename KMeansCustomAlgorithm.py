#custom algorithm built from scratch

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],
	[1.5,1.8],
	[5,8],
	[8,8],
	[1,0.6],
	[9,11]])

colors =10*["g","r","c","b","k"]
# plt.scatter(X[:,0],X[:,1],s=150)
# plt.show()

class K_means:
	def __init__ (self,k=2,tol=0.001,max_iter=2):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter
		
	def fit(self,data):
		self.centroids ={}
		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			self.classification = {}

			for i in range(self.k):
				self.classification[i] =[]

			for featureset in data:
				diatances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
				classification =diatances.index(min(diatances))
				self.classification[classification].append(featureset)

			prev_centroid =dict(self.centroids)
			for classification in self.classification:
				self.centroids[classification] = np.average(self.classification[classification],axis =0 )

			optimized = True

			for c in self.centroids:
				original_centroid =prev_centroid[c]
				current_centroid =self.centroids[c]
				if np.sum((current_centroid - original_centroid)/original_centroid *100.0) > self.tol:
					optimized = False 

			if optimized:
				break


	def predict(self,data):
		diatances = [np.linalg.norm(data - self.centroids[centroid]) for centroidin in self.centroids]
		classification =diatances.index(min(diatances))
		return classification

clf = K_means()

clf.fit(X)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker ="o",color="k",s=150,linewidths=5)

for classification in clf.classification:
	color = colors[classification]
	for featureset in clf.classification[classification]:
		plt.scatter(featureset[0],featureset[1],marker="x",color =color,s=150,linewidths=5)

unknowns = np.array([[1,3],[8,9],[0,3],[5,6],])

for unknown in unknowns:
	classification =clf.predict(unknown)
	plt.scatter(unknown[0],unknown[1],marker="*",color = colors[classification],s=150,linewidths=5)


plt.show()




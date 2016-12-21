#KnearestneighboursAlgorith
#knearest neighbour depends on te Euclidean Distance it is the distance between two data points
'''
Formula Eucliden distance squre root of [sigma n where i=1 * (Qi - pi)**2]

example if q =(1,2) and p =(3,5)
q = co ordinates for data point 
p = quardinates in 2 dymination the points of x and y 

squre root of (1-3)+(2-5)
Euclidean_diatance = sqrt((plot1[0] -plot2[0])**2+(plot1[1]-plot2[1])**2)
'''
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt 
import warnings 
from collections import Counter 
from matplotlib import style
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,4],[7,7],[8,6]]}
new_features = [6,3]



def k_nearest_neighbour(data,predict,k=3):
	if len(data) >= k:
		warnings.warn('k is less than value less than total voting group')

	distance = []
	for group in data:
		for features in data[group]:
			# Euclidean_diatance = np.sqrt(np.sum((np.array(features))**2 - (np.array(predict))**2)) 
			Euclidean_diatance = np.linalg.norm(np.array(features)- np.array(predict)) # faster than  the above line 
			distance.append([Euclidean_diatance,group])

	votes = [i[1] for i in sorted(distance)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

results = k_nearest_neighbour(dataset,new_features)
print(results)

for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1],s=100,color=i)

plt.scatter(new_features[0],new_features[1],s=100,color =results)
plt.show()
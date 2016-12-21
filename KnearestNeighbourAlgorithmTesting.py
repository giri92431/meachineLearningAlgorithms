# KnearestNeighbourAlgorithmTesting.py

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt 
import warnings 
from collections import Counter 
import pandas as pd
import random

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
	confidence  = (Counter(votes).most_common(1)[0][0] /k)
	# print (vote_result,confidence)
	return vote_result ,confidence

# results = k_nearest_neighbour(dataset,new_features)


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace =True)
df.drop(['id'],1,inplace =True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.4
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1]) # [:-1] list up to the last elemet 
for i in test_data:
	test_set[i[-1]].append(i[:-1]) 

correct = 0 
total = 0 
for group in test_set:
	for data in test_set[group]:
		vote, confidence = k_nearest_neighbour(train_set,data,k=5)
		if group ==vote:
			correct +=1
		else:
			print("Confidence",confidence)
		total +=1

print(correct/total)
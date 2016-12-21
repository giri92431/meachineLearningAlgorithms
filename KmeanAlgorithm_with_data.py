import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd


df = pd.read_csv('titanic.csv')
df.drop(['body','name'],1,inplace=True)

df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)



### import code for converting string in to number 
def handel_non_numeric_dataframe(df):
	columns = df.columns.values
	
	for column in columns:
		text_digit_value = {}
		def convert_to_int(val):
			return text_digit_value[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents =df[column].values.tolist()
			unique_elements =set(column_contents)
			x = 0

			for unique in unique_elements:
				if unique not in text_digit_value:
					text_digit_value[unique] =x
					x+=1

			df[column] = list(map(convert_to_int,df[column]))

	return df

df = handel_non_numeric_dataframe(df)

# print(df.head()) 
x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])
clf =KMeans(n_clusters=2)
clf.fit(x)

correct = 0
for i in range(len(x)):
	predict_me = np.array(x[i].astype(float))
	predict_me =predict_me.reshape(-1,len(predict_me))
	prediction = clf.predict(predict_me)
	if prediction[0] == y[i]:
		correct += 1

print(correct/len(x))


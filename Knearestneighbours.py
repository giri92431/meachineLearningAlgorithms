# K nearest neighbours 
# when ever u enter a data point u check how close is the data to nearest elements 
# normally u will keep the value of k =3 for a 2 dimanison array 
# and k = 5 for 3 dimanision

# connfidence and accuriecy are diffent 


# confidence is how confidencend is the data fomn the nearest neighbour 
# ex : k =3 and data are --+ then the data point is 66% confidence it will be near to -- 

import numpy as np
from sklearn import preprocessing ,cross_validation ,neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace =True)
df.drop(['id'],1,inplace =True)

X =np.array(df.drop(['class'],1))
y =np.array(df['class'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

accuriecy = clf.score(x_test,y_test)
print(accuriecy)
exmple_measuew = np.array([3,4,5,4,4,2,2,4,1])
exmple_measuew =exmple_measuew.reshape(1,-1)
predict = clf.predict(exmple_measuew)
print(predict)

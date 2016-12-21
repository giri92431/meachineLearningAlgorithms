# SupportVectorMeachine.py
# find the best serperating hyper plane 


import numpy as np
from sklearn import preprocessing ,cross_validation ,svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace =True)
df.drop(['id'],1,inplace =True)

X =np.array(df.drop(['class'],1))
y =np.array(df['class'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)


clf =svm.SVC()
clf.fit(x_train,y_train)

accuriecy = clf.score(x_test,y_test)
print(accuriecy)
exmple_measuew = np.array([4,2,1,1,1,2,3,1,1])
exmple_measuew =exmple_measuew.reshape(1,-1)
predict = clf.predict(exmple_measuew)
print(predict)

#LinearRegression
###############################algorith##########################
# y =  mx+b

# m (slope )
# b = y (intersept) or best fit

# m = [(mean(x) * mena(y)) - mean(x * y) ]/mean (x) power2 - mean (x power2)

# b = mean(y) - m * mean(x)

###################################Accruecy determinaintion #################
#THE way to determine the accurecy is R^2 or the coefficent or determiontaion 
# it is calculated by squred error
# theroy the error is the distance from the point from the best fit line 
#the reson why  we squre is the we might get negative value when are determining the points from the best fit line 
# we only want to deal with positive values and we also want to avoid out lier

# r**2 = 1 - (SE * y^ )/SE * mean(y)

# SE = squred error
# y = normal y values
# y^ = y hatline or best fit line = b =mean(y) - m * mean(x)



from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import random

# xs = np.array([1,2,3,4,5,6],dtype = np.float64)
# ys = np.array([5,4,6,5,6,7],dtype = np.float64)

def create_dataset(howmany,variance,step=2 ,correlation =False):
	val =1 
	ys= []
	for i in range(howmany):
		y = val + random.randrange(-variance,variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val +=step
		elif correlation and correlation == 'neg':
			val -=step
	xs = [i for i in range(len(ys))]	

	return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
	m = ( ((mean(xs)*mean(ys)) - mean(xs*ys))/ 
		((mean(xs)**2) - mean(xs**2)) )
	b= (mean(ys) - (m * mean(xs)))
	return m , b

def squred_error(ys_orig,ys_line):
	return sum((ys_line - ys_orig)**2)

def coeffienet_of_determination(ys_orig,ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squred_error_reg = squred_error(ys_orig,ys_line)
	squred_error_y_mean = squred_error(ys_orig,y_mean_line)
	return 1 - (squred_error_reg/squred_error_y_mean)




xs,ys =create_dataset(40,40,2,correlation='neg')
m,b =best_fit_slope_and_intercept(xs,ys)

# regession_line = [(m*x)+b for x in xs] 
regession_line = []
for x in xs:
    regession_line.append((m*x)+b)
predict_x = 15
predicrt_y = (m*predict_x)+b
# print (predicrt_y)

r_squred =coeffienet_of_determination(ys,regession_line)
print (r_squred)


plt.scatter(xs,ys)
plt.scatter(predict_x,predicrt_y,color='red')
plt.plot(regession_line)
plt.show()




 
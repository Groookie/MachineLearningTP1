from sklearn import*
from sklearn.preprocessing import scale
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

##Question 1
iris = datasets.load_iris()

print("number of data:")
print (len(iris.data))
print("number of variable:")
print (len(iris.target))

print("iris' data：")
print(iris.data, "\n")
print("iris' target：")
print(iris.target, "\n")
print("iris' feature_names：")
print(iris.feature_names,"\n")
print("iris' target_names：")
print(iris.target_names,"\n")

##Question 2
print("let's make a matrix X")
array_a = np.array([[1,-1,2],[2,0,0],[0,1,-1]])
print(array_a)
print("mean of the matrix X equals:")
array_mean = np.mean(array_a)
print(array_mean)
print("variance of the matrix X equals:")
array_var = np.var(array_a)
print(array_var)

array_a_scaled = scale(array_a)
print("Standardization matrix X is like:")
print(array_a_scaled)

print(np.mean(array_a_scaled), "\n")
print(np.var(array_a_scaled), "\n")
print("Standardization ==> variance = 1")


##Question C
print("let's make a matrix X2")
array_b = np.array([[1,-1,2],[2,0,0],[0,1,-1]])
print(array_b)
print("mean of the matrix X2 equals:")
print(np.mean(array_b))
print("variance of the matrix X2 equals:")
print(np.var(array_b))

min_max_scaler = preprocessing.MinMaxScaler()
print(min_max_scaler)
array_b_minmax = min_max_scaler.fit_transform(array_b)
print("matrix X2 after Normalization")
print(array_b_minmax)
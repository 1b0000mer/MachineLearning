import os
import matplotlib.pyplot as plt
from itertools import cycle
import function as f
import csv
import pandas as pd

#initialization for generate samples data
#dim = 2; 'feature'
#N_data = 50
#N_groups = 3
#max_val = [2, 1, 2, 3, 2, 2]; 'must same as group'
#min_val = [1, 2, 2, 4, 1, 2];

##main: create samples data
#data = f.GenerateData(dim, N_data, N_groups, max_val, min_val)
#kq = f.kmeans(data[0], dim, N_groups, max_val, min_val)
#f1 = f.F1_score(data[1], kq[2])

#print('Number of data: %d' % len(data[0]))
#print('Number of clusters: %d' % len(kq[0]))
#print('F1 Score: %.4f' % f1)

##plot data
#plt.figure(1)
#plt.title('Original')
#j=0
#for i in range(N_groups*N_data):
#    plt.plot(data[0][i][j], data[0][i][j+1], '.')
#    j=0

#plt.figure(2)
##plot data
#j=0
#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#for i, col in zip(range(len(kq[0])), colors):
#    for q in range(len(kq[0][i])):
#        plt.plot(kq[0][i][q][j], kq[0][i][q][j+1], col + '.')
#    j=0

##plot centers
#n=0
#for m, col in zip(range(N_groups), colors):
#    if (len(kq[1][m]) == 0):
#        plt.plot(0, 0, col + '^', markersize=14)
#    else:
#        plt.plot(kq[1][m][n], kq[1][m][n+1], col + '^', markersize=14)
#    n=0

#plt.title('Number of data: %d. Number of clusters: %d. F1 Score: %.4f' % (len(data[0]), len(kq[0]), f1))
#plt.show()
#======================================================================================
##main: read data from csv file
filePath = os.getcwd() + '\data\Mall_Customers_edited.csv'
feature = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
data = pd.read_csv(filePath, names=feature)
print('Done read csv file')
print(data.head())
print('-----------------------------------------------------------------------')
data = data.drop('CustomerID', 1)
x = data.values

N_data = len(x)
dim = x.shape[1]
N_groups = 3

data_out, centers, lables = f.kmeans(x, dim, N_groups, ran_centers=False)
#f1 = f.F1_score(data[1], kq[2])

print('Number of data: %d' % N_data)
print('Number of clusters: %d' % N_groups)
print('Centers:')
print(centers)
#print('F1 Score: %.4f' % f1)
c=0
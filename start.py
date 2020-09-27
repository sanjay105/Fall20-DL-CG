from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy 
A= numpy.loadtxt('datasets-part1/tictac_single.txt')
X= A[:,:9]
y= A[:,9:]
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.7)
X_train=X
y_train=y
# nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X_train)
# print (nbrs)

neigh = KNeighborsRegressor(n_neighbors=9)
y_=neigh.fit(X_train, y_train).predict(X_test)
#y_test_pred = neigh.predict(X_test)
#scr=neigh.score(X_test,y_test)
n_neighbors = 9

knn_unif = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
knn_dist = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_ = knn_unif.fit(X_train, y_train).predict(X_test)
plt.subplot(2, 1,1)
plt.plot(y_test, y_, color='navy', label='prediction')
y_ = knn_dist.fit(X_train, y_train).predict(X_test)

plt.subplot(2, 1,2)
plt.plot(y_test, y_, color='orange', label='prediction')

plt.tight_layout()
#plt.show()
print(knn_dist.score(X_test,y_test))
print(knn_unif.score(X_test,y_test))
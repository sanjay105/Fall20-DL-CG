from sklearn import neighbors
import numpy 
import random
from math import floor

def load_data(fname):
    A= numpy.loadtxt(fname)
    X= A[:,:9]
    y= A[:,9:]
    return X,y

def train(X_train,y_train,n_neighbors):
    knn_dist = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    return knn_dist.fit(X_train,y_train)

def predict(model,X_test):
    return model.predict(X_test)


def __main__():
    X,y=load_data('datasets-part1/tictac_single.txt')
    model = train(X,y,9)
    game(model)
    

def print_board(a):
    print(a[0],"|",a[1],"|",a[2])
    print("____________")
    print(a[3],"|",a[4],"|",a[5])
    print("____________")
    print(a[6],"|",a[7],"|",a[8])

def game(model):
    a=[[0,0,0,0,0,0,0,0,0]]
    if random.randrange(1)==0:
        print("Computer Starts:\n")
        for i in range(5):
            a[0][round(model.predict(a)[0][0])]=-1
            print_board(a[0])
            ind = int(input("Your Move \nenter index to insert:"))
            a[0][ind]=1
    else:
        for i in range(5):
            ind = int(input("Your Move \nenter index to insert:"))
            a[0][ind]=1
            a[0][round(model.predict(a)[0][0])]=-1
            print_board(a[0])
            


__main__()
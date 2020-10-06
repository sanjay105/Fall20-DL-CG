import pandas as pd
import numpy as np
import math
import scipy
from sklearn.model_selection import train_test_split

def load_data(fname):
    A= np.loadtxt(fname)
    X= A[:,:9]
    y= A[:,9:]
    return X,y


X,y=load_data('datasets-part1/tictac_single.txt')

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=5,shuffle=True, test_size=0.2)

X=X_train
y=y_train

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta)
y_pred=X_train.dot(theta)
print(y_pred)
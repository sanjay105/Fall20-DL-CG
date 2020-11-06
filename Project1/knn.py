import pandas as pd
import numpy as np
# imparts for up for learning and evaluation
import math
import scipy
# import some data and classifiers to play with
from sklearn import neighbors
from sklearn import svm
# import some validation tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix,classification_report
import warnings


classification_neighbors=1
regression_neighbors=9


def load_data(fname):
    A= np.loadtxt(fname)
    X= A[:,:9]
    y= A[:,9:]
    return X,y

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true,y_pred,normalize='true')

def get_cross_val_score(model,X,y,cross_val):
    return cross_val_score(model,X,y,cv=cross_val)
    
def fit_model(model,X,y):
    return model.fit(X,y)

def predict_data(model,X_test):
    return model.predict(X_test)

def perform_classification(fname,get_c_matrix=True):
    n_neighbors=classification_neighbors
    examples, labels=load_data(fname)
    X_train, X_test, y_train, y_test = train_test_split(examples, labels,random_state=5,shuffle=True, test_size=0.2)
    nn = neighbors.KNeighborsClassifier(n_neighbors, metric='euclidean')
    model = fit_model(nn,X_train,y_train)
    y_pred=predict_data(model,X_test)
    print("*"*100)
    print("KNN Classification for ",fname," with ",n_neighbors," neighbors:")
    print("Accuracy: ",model.score(X_test,y_test))
    if get_c_matrix:
        print("Confusion Matrix:")
        print(get_confusion_matrix(y_test,y_pred))
    print("*"*100)
    return model

def perform_regression(fname):
    n_neighbors=regression_neighbors
    examples, labels=load_data(fname)
    X_train, X_test, y_train, y_test = train_test_split(examples, labels,random_state=5,shuffle=True, test_size=0.2)
    knn_dist = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    model = fit_model(knn_dist,X_train,y_train)
    y_pred=predict_data(model,X_test)
    print("*"*100)
    print("KNN Regression for ",fname," with ",n_neighbors," neighbors:")
    print("Accuracy: ",model.score(X_test,y_test))
    print("*"*100)
    return model

def knn():
    warnings.filterwarnings('ignore')
    knn_single_classifier_model=perform_classification('datasets-part1/tictac_single.txt',False)
    knn_final_classifier_model=perform_classification('datasets-part1/tictac_final.txt',False)
    knn_multi_classifier_model=perform_classification('datasets-part1/tictac_multi.txt',False)
    knn_multi_regressor_model=perform_regression('datasets-part1/tictac_multi.txt')
    return [knn_single_classifier_model,knn_multi_classifier_model,knn_final_classifier_model,knn_multi_regressor_model]

def main():
    print(knn())


if __name__ == "__main__":
    main()

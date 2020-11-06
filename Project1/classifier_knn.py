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
    n_neighbors=1
    examples, labels=load_data(fname)
    X_train, X_test, y_train, y_test = train_test_split(examples, labels,random_state=5,shuffle=True, test_size=0.2)
    nn = neighbors.KNeighborsClassifier(n_neighbors, metric='euclidean')
    model = fit_model(nn,X_train,y_train)
    y_pred=predict_data(model,X_test)
    #y_pred=y_pred.reshape(-1,1)
    #print("*"*100)
    print("KNN Classification for ",fname,":")
    print("Accuracy: ",model.score(X_test,y_test))
    if get_c_matrix:
        print("Confusion Matrix:")
        print(get_confusion_matrix(y_test,y_pred))
    #print("*"*100)
    return model

def perform_regression(fname):
    n_neighbors=9
    examples, labels=load_data(fname)
    X_train, X_test, y_train, y_test = train_test_split(examples, labels,random_state=5,shuffle=True, test_size=0.2)
    knn_dist = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    model = fit_model(knn_dist,X_train,y_train)
    y_pred=predict_data(model,X_test)
    #print("*"*100)
    print("KNN Regression for ",fname,":")
    print("Accuracy: ",model.score(X_test,y_test))
    # print("Confusion Matrix:")
    # print(get_confusion_matrix(y_test,y_pred))
    #print("*"*100)
    return model

def knn():
    warnings.filterwarnings('ignore')
    knn_single_classifier_model=perform_classification('datasets-part1/tictac_single.txt')
    knn_final_classifier_model=perform_classification('datasets-part1/tictac_final.txt')
    knn_multi_classifier_model=perform_classification('datasets-part1/tictac_multi.txt',False)
    knn_multi_regressor_model=perform_regression('datasets-part1/tictac_multi.txt')
    return [knn_single_classifier_model,knn_multi_classifier_model,knn_final_classifier_model,knn_multi_regressor_model]

def main():
    # examples_single, labels_single = load_data('datasets-part1/tictac_single.txt')
    # examples_multi, labels_multi = load_data('datasets-part1/tictac_multi.txt')
    # examples_final, labels_final= load_data('datasets-part1/tictac_final.txt')
    warnings.filterwarnings('ignore')
    knn()



# def main():
#     examples_single, labels_single = load_data('datasets-part1/tictac_single.txt')
#     examples_multi, labels_multi = load_data('datasets-part1/tictac_multi.txt')
#     rkf = RepeatedKFold(n_splits = 5, n_repeats = 5)
#     nn = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
#     cross_val_score_single = get_cross_val_score(nn,examples_single,labels_single,rkf)
#     cross_val_score_multi = get_cross_val_score(nn,examples_multi,labels_multi,rkf)
#     classification_report()
#     print(cross_val_score_single)
#     print(cross_val_score_multi)

if __name__ == "__main__":
    main()

# examples, labels=load_data('datasets-part1/tictac_single.txt')
# labels=np.ravel(labels)
# rkf = RepeatedKFold(n_splits = 5, n_repeats = 5)
# print(rkf)
# nn = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
# svmRbf = svm.SVC(kernel='rbf', gamma='auto')

# nnScores = cross_val_score(nn, examples, labels, cv = rkf)
# svmScores = cross_val_score(svmRbf, examples, labels, cv = rkf)

# print('Nearest Neighbor: mean=', nnScores.mean(), ', stdDev =', nnScores.std())
# print('RBF SVM: mean=', svmScores.mean(), ', stdDev =', svmScores.std())
# print('Difference between means:', svmScores.mean() - nnScores.mean())
# scipy.stats.mannwhitneyu(nnScores, svmScores)

# X_train, X_test, y_train, y_test = train_test_split(examples, labels,stratify=labels, test_size=0.7)
# knn_model=nn.fit(X_train,y_train)
# y_pred=knn_model.predict(X_test)
# print(knn_model.score(y_test,y_pred))
# a=confusion_matrix(y_test,y_pred)
# k=[]
# ksum=0
# for i in range(9):
#     tsum=np.sum(a[i])
#     ksum = ksum + (tsum-a[i][i])/tsum
#     k.append((tsum-a[i][i])/tsum)
# print(a)
# print(k,ksum/9)
# Importing Required Packages
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, StratifiedKFold 
from sklearn.metrics import confusion_matrix,classification_report
import warnings

from gameplay import game


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
    
def fit_model(model,X,y,isMulti):
    if isMulti:
        model.fit(X,y)
    else:
        skf = StratifiedKFold(n_splits=10, random_state = 777, shuffle=True)
        for train_indices, test_indices in skf.split(X, y):
            model.fit(X[train_indices], y[train_indices])
    return model

def predict_data(model,X_test):
    return model.predict(X_test)

def perform_classification(fname):
    isMulti=False
    if "multi" in fname:
        isMulti=True
    n_neighbors=classification_neighbors
    examples, labels=load_data(fname)
    X_train, X_test, y_train, y_test = train_test_split(examples, labels,random_state=5,shuffle=True, test_size=0.2)
    nn = neighbors.KNeighborsClassifier(n_neighbors, metric='euclidean')
    model = fit_model(nn,X_train,y_train,isMulti)
    y_pred=predict_data(model,X_test)
    # print("*"*100)
    print("KNN Classification for ",fname," with ",n_neighbors," neighbors:")
    print("Accuracy: ",model.score(X_test,y_test))
    if not isMulti:
        print("Confusion Matrix:")
        print(get_confusion_matrix(y_test,y_pred))
    # print("*"*100)
    return model

def perform_regression(fname):
    isMulti=False
    if "multi" in fname:
        isMulti=True
    n_neighbors=regression_neighbors
    examples, labels=load_data(fname)
    X_train, X_test, y_train, y_test = train_test_split(examples, labels,random_state=5,shuffle=True, test_size=0.2)
    knn_dist = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    model = fit_model(knn_dist,X_train,y_train,isMulti)
    y_pred=predict_data(model,X_test)
    # print("*"*100)
    print("KNN Regression for ",fname," with ",n_neighbors," neighbors:")
    print("Accuracy: ",model.score(X_test,y_test))
    # print("*"*100)
    return model

def knn():
    warnings.filterwarnings('ignore')
    classifier_models={}
    regression_models={}
    classifier_models['single']=perform_classification('datasets-part1/tictac_single.txt')
    classifier_models['multi']=perform_classification('datasets-part1/tictac_multi.txt')
    classifier_models['final']=perform_classification('datasets-part1/tictac_final.txt')
    # knn_single_classifier_model=perform_classification('datasets-part1/tictac_single.txt')
    # knn_final_classifier_model=perform_classification('datasets-part1/tictac_final.txt')
    # knn_multi_classifier_model=perform_classification('datasets-part1/tictac_multi.txt')
    regression_models['single']=perform_regression('datasets-part1/tictac_single.txt')
    regression_models['multi']=perform_regression('datasets-part1/tictac_multi.txt')
    regression_models['final']=perform_regression('datasets-part1/tictac_final.txt')
    # knn_multi_regressor_model=perform_regression('datasets-part1/tictac_multi.txt')
    # knn_single_reg=perform_regression('datasets-part1/tictac_single.txt')
    return classifier_models,regression_models


        

def main():
    classifier_models,regression_models=knn()
    print(classifier_models,regression_models)
    #game(scm)


# if __name__ == "__main__":
#     main()

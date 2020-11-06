#!/usr/bin/env python
# coding: utf-8


""" Importing required libraries """
import pandas as pd
import numpy as np
import sys
from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if not sys.warnoptions:
    import warnings



def load_data(file_name):
    """ Function for loading dataset from a text file and slicing in features and labels"""
    content = np.loadtxt(file_name)
    X = content[:,:9]
    y = content[:,9:]
    return X,y




def print_stars():
    """ Function for better visibility of output """
    print("******************************************************************************************************\n")

def mlr_results(dataset, accuracy, time_taken):
    """ Printing result sets to the terminal """
    print("-----------------------Linear Regressor Results on {}---------------------------\n".format(dataset))
    print("Accuracy for Linear Regressor on {}: {}\n".format(dataset, accuracy*100))
    # print_stars()
    # print("Total time to run the Linear Regressor on {}: {} seconds\n".format(dataset, time_taken))
    print_stars()

def multiple_linear_regression(X, y):
    """ Function for classification of intermediate (multi) data set using multiple linear regression through normal equations """
    # Settings hyperparameters
    random_state = 3
    test_size = 0.20
    shuffle = True
    
    # Initial variables for accuracy and no_of_outputs
    n_outputs = 9
    avg_accuracy = 0
    
    # Creates 9 classifier models for each output vector
    for i in range(n_outputs):
        y_data = y[:,i:i+1]
        
        # Splitting training and testing samples
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, random_state=random_state, test_size=test_size, shuffle=shuffle)
        
        # Transpose of X_train
        X_T = X_train.T
        
        # Product of orginal and transpose of X_train
        product_X_XT = X_T.dot(X_train)
        
        # Inverse of the above product
        X_inv = np.linalg.inv(product_X_XT)
        
        # Product of X_inv and X_transpose
        product_Xinv_XT = X_inv.dot(X_T)
        
        # Theta -> Product of above resultant with y_train
        theta = product_Xinv_XT.dot(y_train)
        theta = theta.reshape(9,)
        
        # y_predictions -> product of theta with X_test
        y_pred = X_test.dot(theta)
        
        # Rounding off the predicted values
        for pred_idx in range(len(y_pred)):
            y_pred[pred_idx] = round(y_pred[pred_idx])
        
        # Accuracy of single vector among 9 outputs
        accuracy = accuracy_score(y_test, y_pred)
        avg_accuracy += accuracy
    
    # Total accuracy averaged out
    avg_accuracy /= n_outputs

    return avg_accuracy




def linear_regressor(data_file):
    """ Function for loading data, building SVM model, predicting values, and displaying results """
    # Loads data into features and labels
    X, y = load_data(data_file)
    
    # Start point time
    start = time()
    accuracy = multiple_linear_regression(X, y)
    # End point time
    end = time()
    
    # Prints the results to the terminal
    mlr_results(data_file, accuracy, (end-start))

def main():
    """ Entry point of the program """
    warnings.filterwarnings("ignore")
    filename = "tictac_multi.txt"
    linear_regressor(filename)
    
#if __name__ == "__main__":
#    main()







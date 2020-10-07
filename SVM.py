#!/usr/bin/env python
# coding: utf-8

#### Code for Tic-Tac-Toe SVM Classifier on Final Board, Intermediate Board (Single), and Intermediate Board (Multi)



""" Importing required libraries """
import pandas as pd
import numpy as np
import sys
from time import time
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

if not sys.warnoptions:
    import warnings




def load_data(file_name):
    """ Function for loading dataset from a text file and slicing in features and labels"""
    content = np.loadtxt(file_name)
    X = content[:,:9]
    y = content[:,9:]
    return X,y




def c_matrix(y_test, predictions):
    """ Function for building confusion matrix using test set labels and predicted values """
    confusion_mtrx = confusion_matrix(y_test, predictions)
    confusion_mtrx = confusion_mtrx/confusion_mtrx.astype(np.float).sum(axis=1)
    return confusion_mtrx

def svm_final(X, y):
    """ Function for classification of final data set using SVM classifier """
    # Setting hyperparameters for the classifier
    random_state = 3
    test_size = 0.20
    shuffle = True
    kernel = 'linear' 
    gamma = 'scale'
    
    # Splitting training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size, shuffle=shuffle)
    
    # Creating and fitting the model
    svm = SVC(kernel=kernel, gamma=gamma)
    svm.fit(X_train, y_train)
    
    # Calculating the accuracy score
    accuracy = svm.score(X_test, y_test)
    
    # Predicting the values of test set and building confusion matrix
    predictions = svm.predict(X_test)
    confusion_mtrx = c_matrix(y_test, predictions)
    
    return accuracy, confusion_mtrx, svm

def svm_single(X, y):
    """ Function for classification of intermediate (single) data set using SVM classifier """
    # Setting hyperparameters for the classifier
    random_state = 3
    test_size = 0.20
    shuffle = True
    kernel = 'rbf' 
    gamma = 'scale'
    probability = True
    decision_function_shape='ovo'
    
    # Splitting training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size, shuffle=shuffle)
    
    # Modifying random state for better accuracy
    random_state=10
    
    # Creating and fitting the model
    svm = SVC(kernel=kernel, gamma=gamma, probability=probability, decision_function_shape=decision_function_shape, random_state=random_state)
    svm.fit(X_train, y_train)
    
    # Calculating the accuracy score
    accuracy = svm.score(X_test, y_test)
    
    # Predicting the values of test set and building confusion matrix
    predictions = svm.predict(X_test)
    confusion_mtrx = c_matrix(y_test, predictions)
    
    return accuracy, confusion_mtrx, svm

def svm_multi(X, y):
    """ Function for classification of intermediate (multi) data set using SVM classifier """
    # Setting hyperparameters for the classifier
    random_state = 3
    test_size = 0.20
    shuffle = True
    kernel = 'rbf' 
    gamma = 'scale'
    probability = True
    decision_function_shape='ovo'
    
    # Initial variables for accuracy and no_of_outputs
    avg_accuracy = 0
    n_outputs = 9
    
    # Creates 9 classifier models for each output vector
    for i in range(n_outputs):
        y_data = y[:,i:i+1]
        
        # Splitting training and testing samples
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, random_state=random_state, test_size=test_size, shuffle=shuffle)
        
        # Creating and fitting the model
        svm = SVC(kernel=kernel, gamma=gamma, probability=probability, decision_function_shape=decision_function_shape, random_state=random_state)
        svm.fit(X_train, y_train)
        
        # Calculating the accuracy score
        accuracy = svm.score(X_test, y_test)
        avg_accuracy += accuracy
    
    # Averaging the accuracy score
    avg_accuracy /= n_outputs
    
    return avg_accuracy, svm

def svm_results(dataset, accuracy, confusion_mtrx, time_taken, multi):
    """ Printing result sets to the terminal """
    print("-----------------------SVM Classifier Results on {}---------------------------\n".format(dataset))
    print("Accuracy for svm classifier on {}: {}\n".format(dataset, accuracy*100))
    print_stars()
    if not multi:
        print("Normalized Confusion Matrix for SVM classifier on {}: \n".format(dataset))
        print(confusion_mtrx, "\n")
        print_stars()
    print("Total time to run the SVM classifier on {}: {} seconds\n".format(dataset, time_taken))
    print_stars()

def print_stars():
    """ Function for better visibility of output """
    print("******************************************************************************************************\n")



def svm_classifier(data_file):
    """ Function for loading data, building SVM model, predicting values, and displaying results """
    # Loads data into features and labels
    X, y = load_data(data_file)
    
    # Initial value for multi output dataset
    multi = False
    
    # Start point time
    start = time()
    if 'final' in data_file :
        accuracy, confusion_mtrx, svm = svm_final(X, y)
    elif 'single' in data_file:
        accuracy, confusion_mtrx, svm = svm_single(X, y)
    else:
        accuracy, svm = svm_multi(X, y)
        multi = True
        confusion_mtrx = 0

    # Printing the results
    print("*"*100)
    print("Accuracy for SVM Classification on ",data_file," dataset: ",accuracy)
    if not multi:
        print("Confusion Matrix:")
        print(confusion_mtrx)
    print("*"*100)

    # End point time
    end = time()
    
    # Prints the results to the terminal
    # svm_results(data_file, accuracy, confusion_mtrx, (end-start), multi)
    return svm

def main():
    """ Entry point of the program """
    warnings.filterwarnings("ignore")
    svm_classifier("datasets-part1/tictac_final.txt")
    svm_classifier("datasets-part1/tictac_single.txt")
    svm_classifier("datasets-part1/tictac_multi.txt")
    
# if __name__ == "__main__":
#    main()






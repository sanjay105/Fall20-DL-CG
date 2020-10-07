import warnings
#importing all classifiers and regressors
from MultiLayerPerceptron import MLPClassification,MLPRegression
from knn import knn_classification, knn_regression
from SVM import svm_classifier
from Linear_Regression import linear_regressor

#main function 
def main():
    warnings.filterwarnings('ignore')
    classifiers={}
    regressors={}
    classifiers['mlp']={}
    print("Classifiers:\nMulti Layer Perceptron (MLP):")
    #Performing MLP classification on all datasets
    classifiers['mlp']['single'] = MLPClassification('single')
    classifiers['mlp']['multi'] = MLPClassification('multi')
    classifiers['mlp']['final'] = MLPClassification('final')
    classifiers['knn']={}
    print("KNN:")
    #Performing KNN classification on all datasets
    classifiers['knn']['single']=knn_classification('datasets-part1/tictac_single.txt')
    classifiers['knn']['multi']=knn_classification('datasets-part1/tictac_multi.txt')
    classifiers['knn']['final']=knn_classification('datasets-part1/tictac_final.txt')
    classifiers['svm']={}
    print("SVM:")
    #Performing SVM classification on all datasets
    classifiers['svm']['single']=svm_classifier('datasets-part1/tictac_single.txt')
    classifiers['svm']['multi']=svm_classifier('datasets-part1/tictac_multi.txt')
    classifiers['svm']['final']=svm_classifier('datasets-part1/tictac_final.txt')
    regressors['mlp']={}
    print("Regressors:\nMulti Layer Perceptron (MLP):")
    #Performing MLP Regression on all datasets
    regressors['mlp']['single'] = MLPRegression('single')
    regressors['mlp']['multi'] = MLPRegression('multi')
    regressors['mlp']['final'] = MLPRegression('final')
    regressors['knn']={}
    print("KNN:")
    #Performing KNN Regression on all datasets
    regressors['knn']['single']=knn_regression('datasets-part1/tictac_single.txt')
    regressors['knn']['multi']=knn_regression('datasets-part1/tictac_multi.txt')
    regressors['knn']['final']=knn_regression('datasets-part1/tictac_final.txt')
    #Performing Linear Regression on intermediate board multi data set
    linear_regressor('datasets-part1/tictac_multi.txt')


if __name__ == "__main__":
    main()
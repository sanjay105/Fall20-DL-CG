import pickle
from MultiLayerPerceptron import MLPClassification,MLPRegression
from knn import knn_classification, knn_regression
from SVM import svm_classifier

model = open('gamemodel.mdl','wb')

inp = int(input("Enter \n1.MLP Model\n2.KNN Model \n3.SVM Model\n"))

if inp==1:
    pickle.dump(MLPClassification('single'),model)
elif inp==2:
    pickle.dump(knn_classification('datasets-part1/tictac_single.txt'),model)
elif inp==3:
    pickle.dump(svm_classifier('datasets-part1/tictac_single.txt'),model)

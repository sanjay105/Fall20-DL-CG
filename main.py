from MultiLayerPerceptron import MLPClassification,MLPRegression
from knn import perform_classification, perform_regression








def main():
    classifiers={}
    classifiers['mlp']={}
    print("Classifiers:\nMulti Layer Perceptron (MLP):")
    classifiers['mlp']['single'] = MLPClassification('single')
    classifiers['mlp']['multi'] = MLPClassification('multi')
    classifiers['mlp']['final'] = MLPClassification('final')
    classifiers['knn']={}
    print("KNN:")
    classifiers['knn']['single']=perform_classification('datasets-part1/tictac_single.txt')
    classifiers['knn']['multi']=perform_classification('datasets-part1/tictac_multi.txt')
    classifiers['knn']['final']=perform_classification('datasets-part1/tictac_final.txt')
    classifiers['svm']={}
    print("MLP:")



if __name__ == "__main__":
    main()
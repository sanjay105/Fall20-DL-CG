import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
np.set_printoptions(suppress=True)
np.random.seed(1234)



def read_data(input_file_path, ftype):
    '''
    This method reads the file from <input_file_path> and
    returns X and Y where X is feature vector and Y is class variable
    '''
    data = np.loadtxt(input_file_path)
    if ftype == 'single' or ftype == 'final':
        X, y = np.split(data,[-1],axis=1)
    else:
        X, y = np.split(data,[-9],axis=1)
    return X,y


def MLPClassification(ftype = None):
    '''
    :param ftype: file type to read. ftype takes {'single', 'multi', 'final}
    :return: a classifier that is trained using StratifiedKfold cross validation
    '''
    X,y = read_data('datasets-part1/tictac_'+ftype+'.txt', ftype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if ftype == 'single':
        skf = StratifiedKFold(n_splits=10, random_state = 777, shuffle=True)
        clf = MLPClassifier(solver='adam', alpha=1e-6, max_iter=300, hidden_layer_sizes=(256,256,128,), random_state=777, activation = 'relu')
        for train_indices, test_indices in skf.split(X_train, y_train):
            clf.fit(X[train_indices], np.ravel(y[train_indices], order = 'C'))
            #print(clf.score(X[test_indices], y[test_indices]))
        y_pred = clf.predict(X_test)
        print("Classification Accuracy : " + str(accuracy_score(y_test, y_pred)))
        C = confusion_matrix(y_test, y_pred)
        C = C.astype(np.float) / C.astype(np.float).sum(axis=1)
        print("Confusion Matrix: ")
        print(C)
        return clf
    elif ftype == 'multi':
        clf = MLPClassifier(solver='adam', alpha=1e-6, max_iter=300, hidden_layer_sizes=(256,256,128,), random_state=777, activation = 'relu')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Classification Accuracy : " + str(accuracy_score(y_test, y_pred)))
        return clf
    elif ftype == 'final':
        skf = StratifiedKFold(n_splits=10, random_state = 777, shuffle=True)
#         print(skf.get_n_splits(X, y))
        clf = MLPClassifier(solver='adam', alpha=1e-6, max_iter=300, hidden_layer_sizes=(256,256,128,), random_state=777, activation = 'relu')
        for train_indices, test_indices in skf.split(X_train, y_train):
            clf.fit(X[train_indices], np.ravel(y[train_indices], order = 'C'))
            #print(clf.score(X[test_indices], y[test_indices]))
        y_pred = clf.predict(X_test)
        print("Classification Accuracy : " + str(accuracy_score(y_test, y_pred)))
        print("Confusion Matrix: ")
        C = confusion_matrix(y_test, y_pred)
        #Need to fix this
        C = C.astype(np.float)/C.astype(np.float).sum(axis=1)
        print(C)
        return clf
    return None


def MLPRegression(ftype = None):
    '''

    :param ftype: file type to read. ftype takes {'single', 'multi', 'final}
    :return: a regressor that is trained using StratifiedKfold cross validation
    '''
    X,y = read_data('datasets-part1/tictac_'+ftype+'.txt', ftype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if ftype == 'single' or ftype == 'final':
        skf = StratifiedKFold(n_splits=10, random_state = 777, shuffle=True)
        regr = MLPRegressor(solver='adam', alpha=1e-6, max_iter=300, hidden_layer_sizes=(256,256,128,9), random_state=777, activation = 'relu')
        for train_indices, test_indices in skf.split(X_train, y_train):
            regr.fit(X[train_indices], np.ravel(y[train_indices], order = 'C'))
        y_pred = regr.predict(X_test)
        print("Regressor score (R^2) : ")
        print(regr.score(X_test, y_test))
        return regr
    else:
        regr = MLPRegressor(solver='adam', alpha=1e-6, max_iter=300, hidden_layer_sizes=(256,256,128,9), random_state=777, activation = 'relu')
        regr.fit(X_train, y_train)
        print("Regressor score (R^2) : ")
        print(regr.score(X_test, y_test))
        return regr
    return None


def adjustRegressorValues(y_pred, ftype = None):
    if ftype == 'final':
        y_pred[y_pred<0] = -1
        y_pred[y_pred>=0] = 1
    elif ftype == 'single':
        y_pred[y_pred<=0] = 0
        y_pred[y_pred>=8] = 8
        y_pred = np.floor(y_pred+.5)
    else:
        y_val = np.zeros((y_pred.shape[0], 1))
        for i,vals in enumerate(y_pred):
            mx, idx = -10000000, -1
            for j in range(len(vals)):
                if mx<vals[j]:
                    mx = vals[j]
                    idx = j
            y_val[i] = idx
        y_pred = y_val
    return y_pred


def main():
    mlp_classfier = {}
    mlp_regressor = {}

    mlp_classfier['single'] = MLPClassification('single')
    mlp_regressor['single'] = MLPRegression('single')

    mlp_classfier['multi'] = MLPClassification('multi')
    mlp_regressor['multi'] = MLPRegression('multi')


    mlp_classfier['final'] = MLPClassification('final')
    mlp_regressor['final'] = MLPRegression('final')

    return mlp_classfier, mlp_regressor



'''
    File name: chessClassifier.py
    Author: Steven Haertling
    Date created: 4/24/2021
    Date last modified: 4/24/2021
    Python Version: 3.8.6
'''


#imports
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import graphviz
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def processKRKP( ):
    # read from file
    S = np.genfromtxt("data/kr-vs-kp.data", missing_values=0, skip_header=0, delimiter=',', dtype=str)
    matrix = np.empty_like(S[:,0])
    for i in range(len(S[0])):
        enc = preprocessing.LabelEncoder()
        col_vector = enc.fit_transform(S[:,i])
        matrix = np.c_[matrix, col_vector]
    # make a test set out of the complete data set
    testSize = round(len(matrix[:, 37])*.7)
    x_train, x_test, y_train, y_test = train_test_split( matrix[:, 1:37], matrix[:, 37], test_size=testSize, random_state=4 )
    return x_train, x_test, y_train, y_test


def processKRK( S ):

    enc = preprocessing.OrdinalEncoder()
    # X = [['a', 1],['b',2],['c',3],['d',4],['e',5],['f',6],['g',7],['h',8]]
    # enc.fit(X)
    matrix = enc.fit_transform(S[:, 0:6])
    le = preprocessing.LabelEncoder()
    col_vector = le.fit_transform(S[:,6])
    # np.set_printoptions(threshold=sys.maxsize)
    # print(S[:,6])
    # print(col_vector)
    return matrix, col_vector

def decision_tree_driverKRvKP( x_train, x_test, y_train, y_test ):
    
    # run the classifer a bunch with didfferent parameters 
    depths = [1,3,5,10,20,30,40,50]
    features = [1,2,3,4,5,10,15]
    for depth in depths:    
        for feature in features:
            d_tree = tree.DecisionTreeClassifier(max_depth=depth,max_features=feature)
            decision_tree = d_tree.fit(x_train, y_train)
            prediction = decision_tree.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            print("max depth = {}, feature splits = {}".format(depth, feature))
            print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            print("accuracy = {}".format(error))

def bagging_dtree_driverKRVKP( x_train, x_test, y_train, y_test ):
    depths = [1,2,3,5,7,10]
    bag_sizes = [1,3,5,10,15]
    for depth in depths:    
        for bag_size in bag_sizes:
            sBag = BaggingClassifier( base_estimator=tree.DecisionTreeClassifier(max_depth=depth, max_features=1),
                                      n_estimators=bag_size).fit( x_train, y_train )
            prediction = sBag.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            print("max depth = {}, bag size = {}".format(depth, bag_size))
            print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            print("accuracy = {}".format(error))

def boosting_dtree_driverKRVKP( x_train, x_test, y_train, y_test ):
    depths = [1,2]
    bag_sizes = [10,20,30,40,50,60]
    for depth in depths:    
        for bag_size in bag_sizes:
            sBoost = AdaBoostClassifier( base_estimator=tree.DecisionTreeClassifier(max_depth=depth, max_features=1),
                                         n_estimators=bag_size)
            sBoost.fit(x_train,y_train)
            prediction = sBoost.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            print("max depth = {}, bag size = {}".format(depth, bag_size))
            print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            print("accuracy = {}".format(error))

# def decision_tree_driverKRvK():
#     #read from file
#     S = np.genfromtxt("data/krkopt.data", missing_values=0, skip_header=0, delimiter=',', dtype=str)
#     for i in range(len(S[0])):
#         values, counts = np.unique (  S[:,i], return_counts=True )
#         print("i = {}, values = {}, counts = {}\n".format( i, values, counts))
#     #change str to ints
#     x, y = processKRK( S )
#     for i in range(len(x[0])):
#         values, counts = np.unique (  x[:,i], return_counts=True )
#         print("i = {}, values = {}, counts = {}\n".format( i, values, counts))
#     values, counts = np.unique (  y, return_counts=True )
#     print("i = {}, values = {}, counts = {}\n".format( "y", values, counts))

if __name__ == '__main__':
    # process data into scikit usable form
    x_train, x_test, y_train, y_test = processKRKP()
    # use data in normal dtree
    decision_tree_driverKRvKP( x_train, x_test, y_train, y_test )
    # use data in bagging dtree
    bagging_dtree_driverKRVKP( x_train, x_test, y_train, y_test )
    # use data in boosting dtree
    boosting_dtree_driverKRVKP( x_train, x_test, y_train, y_test )
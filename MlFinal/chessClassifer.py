'''
    File name: chessClassifier.py
    Author: Steven Haertling
    Date created: 04/24/2021
    Date last modified: 05/05/2021
    Python Version: 3.8.6
'''


#imports
import numpy as np
import sys
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import graphviz
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
g_list = list()

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

def decision_tree_driverKRvKP( x_train, x_test, y_train, y_test ):
    
    # run the classifer a bunch with didfferent parameters 
    depths = [1,3,5,10,20,30,40,50]
    features = [1,2,3,4,5,10,15]
    for depth in depths:    
        for feature in features:
            t0 = time.time()
            d_tree = tree.DecisionTreeClassifier(max_depth=depth,max_features=feature)
            decision_tree = d_tree.fit(x_train, y_train)
            prediction = decision_tree.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, feature, tp,fn,fp,tn,error, runTime, "dTree"]
            g_list.append(myTuple)
            # print("max depth = {}, feature splits = {}, runTime = {}".format(depth, feature, runTime))
            # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            # print("accuracy = {}".format(error))
    # print(g_list)

def bagging_dtree_driverKRVKP( x_train, x_test, y_train, y_test ):
    depths = [1,2,3,5,7,10]
    bag_sizes = [1,3,5,10,15]
    for depth in depths:    
        for bag_size in bag_sizes:
            t0 = time.time()
            sBag = BaggingClassifier( base_estimator=tree.DecisionTreeClassifier(max_depth=depth, max_features=1),
                                      n_estimators=bag_size).fit( x_train, y_train )
            prediction = sBag.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, bag_size, tp,fn,fp,tn,error, runTime, "bagging"]
            g_list.append(myTuple)
            # print("max depth = {}, bag size = {}, runTime = {}".format( depth, bag_size, runTime ))
            # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            # print("accuracy = {}".format(error))

def boosting_dtree_driverKRVKP( x_train, x_test, y_train, y_test ):
    depths = [1,2]
    bag_sizes = [10,20,30,40,50,60]
    for depth in depths:    
        for bag_size in bag_sizes:
            t0 = time.time()
            sBoost = AdaBoostClassifier( base_estimator=tree.DecisionTreeClassifier(max_depth=depth, max_features=1),
                                         n_estimators=bag_size)
            sBoost.fit(x_train,y_train)
            prediction = sBoost.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, bag_size, tp,fn,fp,tn,error, runTime, "boosting"]
            g_list.append(myTuple)
            # print("max depth = {}, bag size = {}, runTime = {}".format(depth, bag_size, runTime))
            # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            # print("accuracy = {}".format(error))

def knn_driver_KRVKP( x_train, x_test, y_train, y_test ):
    weights = ['uniform', 'distance']
    algorithms = ['ball_tree', 'kd_tree', 'brute']
    num_of_neighbors = [1, 2, 3, 5, 10, 15, 30]
    leaf_sizes = [2, 5, 10, 15, 20, 30, 40, 50]
    for alg in algorithms:
        for weight in weights:
            for num_neigh in num_of_neighbors:
                for leaf in leaf_sizes:
                    t0 = time.time()
                    neigh = KNeighborsClassifier( n_neighbors=num_neigh,weights=weight,algorithm=alg,leaf_size=leaf).fit( x_train, y_train )
                    prediction = neigh.predict( x_test )
                    tn, fp, fn, tp = confusion_matrix( y_test, prediction ).ravel()
                    error = accuracy_score( y_test, prediction )
                    t1 = time.time()
                    runTime = t1 - t0
                    myTuple = [alg, weight, num_neigh, leaf, tp,fn,fp,tn,error, runTime, "knn"]
                    g_list.append(myTuple)
                    # print("algorithm = {}, weight = {}, k-nn = {}, leaf = {}, time = {}".format(alg, weight, num_neigh, leaf, runTime))
                    # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
                    # print("accuracy = {}".format(error))

def logReg_driver_KRVKP( x_train, x_test, y_train, y_test ):

    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    Cs = [.001, .01, .1, 1, 10, 100]
    for solver in solvers:
        for C in Cs:
            t0 = time.time()
            model = LogisticRegression( solver=solver, C=C, max_iter=4000 ).fit( x_train, y_train )
            prediction = model.predict( x_test )
            tn, fp, fn, tp = confusion_matrix( y_test, prediction ).ravel()
            error = accuracy_score( y_test, prediction )
            t1 = time.time()
            runTime = t1 - t0
            myTuple = [solver, C, tp,fn,fp,tn,error,runTime,"log"]
            g_list.append(myTuple)
            # print("algorithm = {}, C = {}, time = {}".format(solver, C, runTime))
            # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            # print("accuracy = {}\n".format(error))

if __name__ == '__main__':
    # Decision Trees, Bagging, Boosting, Logistic Regression, KNN
    # process data into scikit usable form
    t0 = time.time()
    x_train, x_test, y_train, y_test = processKRKP()
    # x_train, x_test, y_train, y_test = processKRKP2()
    # # use data in normal dtree
    decision_tree_driverKRvKP( x_train, x_test, y_train, y_test )
    # use data in bagging dtree
    bagging_dtree_driverKRVKP( x_train, x_test, y_train, y_test )
    # use data in boosting dtree
    boosting_dtree_driverKRVKP( x_train, x_test, y_train, y_test )
    # use data in knn
    knn_driver_KRVKP( x_train, x_test, y_train, y_test )
    # use data in logistic regression classifier
    logReg_driver_KRVKP( x_train, x_test, y_train, y_test )
    t1 = time.time()
    runTime = t1 - t0
    print("runTime = {}".format(runTime))
    for myTuple in g_list:
        print(myTuple)
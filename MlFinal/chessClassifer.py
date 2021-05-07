'''
    File name: chessClassifier.py
    Author: Steven Haertling
    Date created: 04/24/2021
    Date last modified: 05/05/2021
    Python Version: 3.8.6
'''


#imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix

import graphviz
import sys
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
    x_train = x_train.astype(int)
    x_test = x_test.astype(int)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    return x_train, x_test, y_train, y_test

def decision_tree_driverKRvKP( x_train, x_test, y_train, y_test ):
    
    # run the classifer a bunch with didfferent parameters 
    depths = [1,3,5,10,20,30,40,50]
    features = [1,2,3,4,5,10,15]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["depth", "feature", "tp", "fn", "fp", "tn", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for depth in depths:    
        i = 0
        for feature in features:
            t0 = time.time()
            d_tree = tree.DecisionTreeClassifier(max_depth=depth,max_features=feature)
            decision_tree = d_tree.fit(x_train, y_train)
            prediction = decision_tree.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            y_score = decision_tree.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("Dtree ROC Curve of Depth = {}".format(depth))
            plt.plot(fpr, tpr, color=colors[i])
            
            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, feature, tp, fn, fp, tn, roc_score, error, runTime, "dTree"]
            g_list.append(myTuple)
            i = i+1
            # print("max depth = {}, feature splits = {}, runTime = {}".format(depth, feature, runTime))
            # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            # print("accuracy = {}".format(error))
        plt.legend(["1 feature split","2 feature split","3 feature split","4 feature split","5 feature split","10 feature split","15 feature split"], loc ="lower right")
        plt.savefig("dtreePlots/ROC-Depth-{}.png".format(depth))

def bagging_dtree_driverKRVKP( x_train, x_test, y_train, y_test ):
    depths = [1,3,5,10,20,30,40,50]
    bag_sizes = [1,3,5,10,15,30]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["depth", "bag_size", "tp", "fn", "fp", "tn", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for depth in depths:
        i = 0
        for bag_size in bag_sizes:
            t0 = time.time()
            sBag = BaggingClassifier( base_estimator=tree.DecisionTreeClassifier(max_depth=depth, max_features=1),
                                      n_estimators=bag_size).fit( x_train, y_train )
            prediction = sBag.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )

            y_score = sBag.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("Bagging ROC Curve of Depth = {}".format(depth))
            plt.plot(fpr, tpr, color=colors[i])

            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, bag_size, tp, fn, fp, tn, roc_score, error, runTime, "bagging"]
            g_list.append(myTuple)
            i = i + 1
            # print("max depth = {}, bag size = {}, runTime = {}".format( depth, bag_size, runTime ))
            # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            # print("accuracy = {}".format(error))
        plt.legend(["1 bags","3 bags","5 bags","10 bags","15 bags","30 bags"], loc ="lower right")
        plt.savefig("baggingPlots/ROC-Depth-{}.png".format(depth))

def boosting_dtree_driverKRVKP( x_train, x_test, y_train, y_test ):
    depths = [1,2]
    bag_sizes = [10,20,30,40,50,60]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["depth", "bag_size", "tp", "fn", "fp", "tn", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for depth in depths:
        i = 0
        for bag_size in bag_sizes:
            t0 = time.time()
            sBoost = AdaBoostClassifier( base_estimator=tree.DecisionTreeClassifier(max_depth=depth, max_features=1),
                                         n_estimators=bag_size)
            sBoost.fit(x_train,y_train)
            prediction = sBoost.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )

            y_score = sBoost.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("Bagging ROC Curve of Depth = {}".format(depth))
            plt.plot(fpr, tpr, color=colors[i])

            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, bag_size, tp,fn,fp,tn,error, runTime, "boosting"]
            g_list.append(myTuple)
            i = i + 1
            # print("max depth = {}, bag size = {}, runTime = {}".format(depth, bag_size, runTime))
            # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            # print("accuracy = {}".format(error))
        plt.legend(["10 bags","20 bags","30 bags","40 bags","50 bags","60 bags"], loc ="lower right")
        plt.savefig("boostingPlots/ROC-Depth-{}.png".format(depth))

def knn_driver_KRVKP( x_train, x_test, y_train, y_test ):
    weights = ['uniform', 'distance']
    algorithms = ['ball_tree', 'kd_tree', 'brute']
    num_of_neighbors = [1, 2, 3, 5, 10, 15, 30]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["alg", "weight", "k-neighbors", "tp", "fn", "fp", "tn", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for alg in algorithms:
        for weight in weights:
            i = 0
            for num_neigh in num_of_neighbors:
                t0 = time.time()
                neigh = KNeighborsClassifier( n_neighbors=num_neigh,weights=weight,algorithm=alg).fit( x_train, y_train )
                prediction = neigh.predict( x_test )
                tn, fp, fn, tp = confusion_matrix( y_test, prediction ).ravel()
                error = accuracy_score( y_test, prediction )

                y_score = neigh.predict_proba( x_test )
                roc_score = roc_auc_score(y_test, y_score[:,1])
                fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
                if i == 0:
                    plt.clf()
                    plt.xlabel('False Positive Ratio')
                    plt.ylabel('True Positive Ratio')
                    plt.title("k-nn ROC Curve: alg = {}, weight = {}".format(alg,weight))
                plt.plot(fpr, tpr, color=colors[i])

                t1 = time.time()
                runTime = t1 - t0
                myTuple = [alg, weight, num_neigh, tp, fn, fp, tn, roc_score, error, runTime, "knn"]
                g_list.append(myTuple)
                i = i + 1
                # print("algorithm = {}, weight = {}, k-nn = {}, time = {}".format(alg, weight, num_neigh, runTime))
                # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
                # print("accuracy = {}".format(error))
                plt.legend(["1-nn", "2-nn", "3-nn", "5-nn", "10-nn", "15-nn", "30-nn"], loc ="lower right")
                plt.savefig("knnPlots/ROC-{}-{}.png".format(alg,weight))
                # plt.show()

def logReg_driver_KRVKP( x_train, x_test, y_train, y_test ):

    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    Cs = [.001, .01, .1, 1, 10, 100]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["solver", "C", "tp", "fn", "fp", "tn", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for solver in solvers:
        i = 0
        for C in Cs:
            t0 = time.time()
            model = LogisticRegression( solver=solver, C=C, max_iter=4000 ).fit( x_train, y_train )
            prediction = model.predict( x_test )
            tn, fp, fn, tp = confusion_matrix( y_test, prediction ).ravel()
            error = accuracy_score( y_test, prediction )

            y_score = model.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("Logistic Regression ROC Curve: Solver = {}".format(solver))
            plt.plot(fpr, tpr, color=colors[i])

            t1 = time.time()
            runTime = t1 - t0
            myTuple = [solver, C, tp, fn, fp, tn, roc_score, error, runTime, "log"]
            g_list.append(myTuple)
            i = i + 1
            # print("algorithm = {}, C = {}, time = {}".format(solver, C, runTime))
            # print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            # print("accuracy = {}\n".format(error))
        plt.legend(["C = .001", "C = .01", "C = .1", "C = 1", "C = 10", "C = 100"], loc ="lower right")
        plt.savefig("logRegPlots/ROC-Solver-{}.png".format(solver))

if __name__ == '__main__':
    # Decision Trees, Bagging, Boosting, Logistic Regression, KNN
    # process data into scikit usable form
    t0 = time.time()
    x_train, x_test, y_train, y_test = processKRKP()
    # x_train, x_test, y_train, y_test = processKRKP2()
    # # use data in normal dtree
    # decision_tree_driverKRvKP( x_train, x_test, y_train, y_test )
    # use data in bagging dtree
    # bagging_dtree_driverKRVKP( x_train, x_test, y_train, y_test )
    # # use data in boosting dtree
    # boosting_dtree_driverKRVKP( x_train, x_test, y_train, y_test )
    # # use data in knn
    knn_driver_KRVKP( x_train, x_test, y_train, y_test )
    # # use data in logistic regression classifier
    # logReg_driver_KRVKP( x_train, x_test, y_train, y_test )
    t1 = time.time()
    runTime = t1 - t0
    print("runTime = {}".format(runTime))
    for myTuple in g_list:
        print(myTuple)
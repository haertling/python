'''
    File name: chessClassifier.py
    Author: Steven Haertling
    Date created: 04/24/2021
    Date last modified: 05/08/2021
    Python Version: 3.8.6
'''


#imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve, precision_recall_curve, average_precision_score

import graphviz
import os
import sys
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
g_list = list()

def confusionMatrixDriver( classifer, x, y , filename, title ):
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix( classifer, x, y, cmap=plt.cm.Greens, ax=ax2)
    ax2.set_title(title)
    fig2.savefig( filename )
    plt.close(fig2)
    # print(filename)

def precisionRecallDriver( classifer, x, y , filename, title ):
    fig3, ax3 = plt.subplots()
    plot_precision_recall_curve( classifer, x, y, ax=ax3)
    ax3.set_title(title)
    fig3.savefig( filename )
    plt.close(fig3)
    # print(filename)

def init():
    if not os.path.exists("KRKP/baggingPlots"):
        os.makedirs("KRKP/baggingPlots")
    if not os.path.exists("KRKP/boostingPlots"):
        os.makedirs("KRKP/boostingPlots")
    if not os.path.exists("KRKP/dtreePlots"):
        os.makedirs("KRKP/dtreePlots")
    if not os.path.exists("KRKP/knnPlots"):
        os.makedirs("KRKP/knnPlots")
    if not os.path.exists("KRKP/logRegPlots"):
        os.makedirs("KRKP/logRegPlots")
    if not os.path.exists("KRKP/mlpPlots"):
        os.makedirs("KRKP/mlpPlots")

def processKRKP( ):
    # read from file
    S = np.genfromtxt("data/kr-vs-kp.data", missing_values=0, skip_header=0, delimiter=',', dtype=str)
    matrix = np.empty_like(S[:,0])
    for i in range(len(S[0])):
        enc = preprocessing.LabelEncoder()
        col_vector = enc.fit_transform(S[:,i])
        matrix = np.c_[matrix, col_vector]
    # make a test set out of the complete data set
    
    x_train, x_test, y_train, y_test = train_test_split( matrix[:, 1:37], matrix[:, 37], test_size=.7, random_state=4 )
    x_train = x_train.astype(int)
    x_test = x_test.astype(int)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    return x_train, x_test, y_train, y_test

def decision_tree_driverKRvKP( x_train, x_test, y_train, y_test ):
    
    depths = [1,3,5,10,20,30,40,50]
    features = [1,2,3,4,5,10,15]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["depth", "feature", "tp", "fn", "fp", "tn", "true positive rate", "false positive rate", "precision recall score", "10-CV accuracy", "10-CV Std Deviation", "roc_auc_score", "accuracy", "runTime", "classifier"]
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
            pr_score = average_precision_score( y_test, prediction )

            scores = cross_val_score( d_tree, x_train, y_train, cv=10 )
            print("{0:.2f} accuracy with a standard deviation of {1:.2f}, depth = {2}, # of features = {3}".format( scores.mean(), scores.std(), depth, feature ))#debug

            confusionMatrixDriver( d_tree, x_test, y_test, "KRKP/dtreePlots/ConfusionMatrix-Depth-{}-features-{}.png".format( depth, feature ), "Depth = {}, number of features ={}".format( depth, feature ) )
            precisionRecallDriver( d_tree, x_test, y_test, "KRKP/dtreePlots/PrecisionRecall-Depth-{}-features-{}.png".format( depth, feature ), "Depth = {}, number of features ={}".format( depth, feature ) )
            y_score = decision_tree.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("Dtree ROC Curve of Depth = {}".format(depth))
            plt.plot(fpr, tpr, color=colors[i])

            truePosRate = tp/(tp+fn)
            falsePosRate = fp/(fp+tn)

            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, feature, tp, fn, fp, tn, truePosRate, falsePosRate, pr_score, scores.mean(), scores.std(), roc_score, error, runTime, "dTree"]
            g_list.append(myTuple)
            i = i+1
        plt.legend(["1 feature split","2 feature split","3 feature split","4 feature split","5 feature split","10 feature split","15 feature split"], loc ="lower right")
        plt.savefig("KRKP/dtreePlots/ROC-Depth-{}.png".format(depth))

def bagging_dtree_driverKRVKP( x_train, x_test, y_train, y_test ):
    depths = [1,3,5,10,20,30,40,50]
    bag_sizes = [1,3,5,10,15,30]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["depth", "bag_size", "tp", "fn", "fp", "tn", "true positive rate", "false positive rate", "precision recall score", "10-CV accuracy", "10-CV Std Deviation", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for depth in depths:
        i = 0
        for bag_size in bag_sizes:
            t0 = time.time()
            sBag = BaggingClassifier( base_estimator=tree.DecisionTreeClassifier(max_depth=depth, max_features=1), n_estimators=bag_size )
            model = sBag
            sBag.fit( x_train, y_train )
            prediction = sBag.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            pr_score = average_precision_score( y_test, prediction )

            scores = cross_val_score( model, x_train, y_train, cv=10 )
            print("{0:.2f} accuracy with a standard deviation of {1:.2f}, depth = {2}, bag_size = {3}".format( scores.mean(), scores.std(), depth, bag_size ))#debug

            confusionMatrixDriver( sBag, x_test, y_test,"KRKP/baggingPlots/ConfusionMatrix-Depth-{}-BS-{}.png".format( depth, bag_size ), "Depth = {}, Bag Size = {}".format( depth, bag_size ) )
            precisionRecallDriver( sBag, x_test, y_test,"KRKP/baggingPlots/PrecisionRecall-Depth-{}-BS-{}.png".format( depth, bag_size ), "Depth = {}, Bag Size = {}".format( depth, bag_size ) )
            y_score = sBag.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("Bagging ROC Curve of Depth = {}".format(depth))
            plt.plot(fpr, tpr, color=colors[i])

            truePosRate = tp/(tp+fn)
            falsePosRate = fp/(fp+tn)

            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, bag_size, tp, fn, fp, tn, truePosRate, falsePosRate, pr_score, scores.mean(), scores.std(), roc_score, error, runTime, "bagging"]
            g_list.append(myTuple)
            i = i + 1
        plt.legend(["1 bags","3 bags","5 bags","10 bags","15 bags","30 bags"], loc ="lower right")
        plt.savefig("KRKP/baggingPlots/ROC-Depth-{}.png".format(depth))

def boosting_dtree_driverKRVKP( x_train, x_test, y_train, y_test ):
    depths = [1,2]
    n_estimators = [10,20,30,40,50,60]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["depth", "bag_size", "tp", "fn", "fp", "tn", "true positive rate", "false positive rate", "precision recall score", "10-CV accuracy", "10-CV Std Deviation", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for depth in depths:
        i = 0
        for n_estimator in n_estimators:
            t0 = time.time()
            sBoost = AdaBoostClassifier( base_estimator=tree.DecisionTreeClassifier(max_depth=depth, max_features=1),
                                         n_estimators=n_estimator)
            model = sBoost
            sBoost.fit(x_train,y_train)
            prediction = sBoost.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            pr_score = average_precision_score( y_test, prediction )

            scores = cross_val_score( model, x_train, y_train, cv=10 )
            print("{0:.2f} accuracy with a standard deviation of {1:.2f}, depth = {2}, estimators = {3}".format( scores.mean(), scores.std(), depth, n_estimator ))#debug

            confusionMatrixDriver( sBoost, x_test, y_test,"KRKP/boostingPlots/ConfusionMatrix-Depth-{}-BS-{}.png".format( depth, n_estimator ), "Depth = {}, Number of estimators = {}".format( depth, n_estimator ))
            precisionRecallDriver( sBoost, x_test, y_test,"KRKP/boostingPlots/PrecisionRecall-Depth-{}-BS-{}.png".format( depth, n_estimator ), "Depth = {}, Number of estimators = {}".format( depth, n_estimator ))
            
            y_score = sBoost.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("Boosting ROC Curve of Depth = {}".format(depth))
            plt.plot(fpr, tpr, color=colors[i])

            truePosRate = tp/(tp+fn)
            falsePosRate = fp/(fp+tn)

            t1 = time.time()
            runTime = t1 - t0
            myTuple = [depth, n_estimator, tp, fn, fp, tn, truePosRate, falsePosRate, pr_score, scores.mean(), scores.std(), roc_score, error, runTime, "boosting"]
            g_list.append(myTuple)
            i = i + 1
        plt.legend(["10 estimators","20 estimators","30 estimators","40 estimators","50 estimators","60 estimators"], loc ="lower right")
        plt.savefig("KRKP/boostingPlots/ROC-Depth-{}.png".format(depth))

def knn_driver_KRVKP( x_train, x_test, y_train, y_test ):
    weights = ['uniform', 'distance']
    algorithms = ['ball_tree', 'kd_tree', 'brute']
    num_of_neighbors = [1, 2, 3, 5, 10, 15, 30]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["alg", "weight", "k-neighbors", "tp", "fn", "fp", "tn", "true positive rate", "false positive rate", "precision recall score", "10-CV accuracy", "10-CV Std Deviation", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for alg in algorithms:
        for weight in weights:
            i = 0
            for num_neigh in num_of_neighbors:
                t0 = time.time()
                neigh = KNeighborsClassifier( n_neighbors=num_neigh,weights=weight,algorithm=alg)
                model = neigh
                neigh.fit( x_train, y_train )
                prediction = neigh.predict( x_test )
                tn, fp, fn, tp = confusion_matrix( y_test, prediction ).ravel()
                error = accuracy_score( y_test, prediction )
                pr_score = average_precision_score( y_test, prediction )

                scores = cross_val_score( model, x_train, y_train, cv=10 )
                print("{0:.2f} accuracy with a standard deviation of {1:.2f}, solver = {2}, weight = {3}, {4}-nn".format( scores.mean(), scores.std(), alg, weight, num_neigh ))#debug

                confusionMatrixDriver( neigh, x_test, y_test, "KRKP/knnPlots/confusionMatrix-{}-{}-{}.png".format( num_neigh, alg, weight ), "alg = {}, weight = {}, neighbors = {}".format( alg, weight, num_neigh ))
                precisionRecallDriver( neigh, x_test, y_test, "KRKP/knnPlots/PrecisionRecall-{}-{}-{}.png".format( num_neigh, alg, weight ), "alg = {}, weight = {}, neighbors = {}".format( alg, weight, num_neigh ))
                y_score = neigh.predict_proba( x_test )
                roc_score = roc_auc_score(y_test, y_score[:,1])
                fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
                if i == 0:
                    plt.clf()
                    plt.xlabel('False Positive Ratio')
                    plt.ylabel('True Positive Ratio')
                    plt.title("k-nn ROC Curve: alg = {}, weight = {}".format( alg, weight ))
                plt.plot(fpr, tpr, color=colors[i])

                truePosRate = tp/(tp+fn)
                falsePosRate = fp/(fp+tn)

                t1 = time.time()
                runTime = t1 - t0
                myTuple = [alg, weight, num_neigh, tp, fn, fp, tn, truePosRate, falsePosRate, pr_score, scores.mean(), scores.std(), roc_score, error, runTime, "knn"]
                g_list.append(myTuple)
                i = i + 1
            plt.legend(["1-nn", "2-nn", "3-nn", "5-nn", "10-nn", "15-nn", "30-nn"], loc ="lower right")
            plt.savefig("KRKP/knnPlots/ROC-{}-{}.png".format(alg, weight))

def logReg_driver_KRVKP( x_train, x_test, y_train, y_test ):

    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    Cs = [.001, .01, .1, 1, 10, 100]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["solver", "C", "tp", "fn", "fp", "tn", "true positive rate", "false positive rate", "precision recall score", "10-CV accuracy", "10-CV Std Deviation", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)
    for solver in solvers:
        i = 0
        for C in Cs:
            t0 = time.time()
            model = LogisticRegression( solver=solver, C=C, max_iter=4000 )
            logModel = model
            model.fit( x_train, y_train )
            prediction = model.predict( x_test )
            tn, fp, fn, tp = confusion_matrix( y_test, prediction ).ravel()
            error = accuracy_score( y_test, prediction )
            pr_score = average_precision_score( y_test, prediction )

            scores = cross_val_score( logModel, x_train, y_train, cv=10 )
            print("{0:.2f} accuracy with a standard deviation of {1:.2f}, solver = {2}, C = {3}".format( scores.mean(), scores.std(), solver, C ))#debug 

            confusionMatrixDriver( model, x_test, y_test,"KRKP/logRegPlots/ConfusionMatrix-Solver-{}-C-{}.png".format(solver,C), "Solver-{} C-{}".format(solver,C))
            precisionRecallDriver( model, x_test, y_test,"KRKP/logRegPlots/PrecisionRecall-Solver-{}-C-{}.png".format(solver,C), "Solver-{} C-{}".format(solver,C))
            y_score = model.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("Logistic Regression ROC Curve: Solver = {}".format(solver))
            plt.plot(fpr, tpr, color=colors[i])

            truePosRate = tp/(tp+fn)
            falsePosRate = fp/(fp+tn)

            t1 = time.time()
            runTime = t1 - t0
            myTuple = [solver, C, tp, fn, fp, tn, truePosRate, falsePosRate, pr_score, scores.mean(), scores.std(), roc_score, error, runTime, "log"]
            g_list.append(myTuple)
            i = i + 1
        plt.legend(["C = .001", "C = .01", "C = .1", "C = 1", "C = 10", "C = 100"], loc ="lower right")
        plt.savefig("KRKP/logRegPlots/ROC-Solver-{}.png".format(solver))

def mlp_driver_KRVKP( x_train, x_test, y_train, y_test ):
    activations = ["identity", "tanh"]
    sizes = [(2, 2), (5, 2), (10, 2), (2, 5), (2, 10), (10, 10), (50, 2)]
    colors = ['b','g','r','c','m','y','k']
    myTuple = ["size", "activation", "tp", "fn", "fp", "tn", "true positive rate", "false positive rate", "precision recall score", "10-CV accuracy", "10-CV Std Deviation", "roc_auc_score", "accuracy", "runTime", "classifier"]
    g_list.append(myTuple)

    for activation in activations:
        i = 0
        for size in sizes:
            t0 = time.time()
            mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=size, random_state=1, activation=activation, max_iter=1000)
            model = mlp
            mlp.fit( x_train, y_train )
            prediction = mlp.predict( x_test )
            tn, fp, fn, tp = confusion_matrix( y_test, prediction ).ravel()
            error = accuracy_score( y_test, prediction )
            pr_score = average_precision_score( y_test, prediction )

            scores = cross_val_score( model, x_train, y_train, cv=10 )
            print("{0:.2f} accuracy with a standard deviation of {1:.2f}, solver = {2}, C = {3}".format( scores.mean(), scores.std(), activation, size ))#debug 

            confusionMatrixDriver( mlp, x_test, y_test,"KRKP/mlpPlots/ConfusionMatrix-{}-{}-{}.png".format(activation, size[0], size[1] ), "Activation {}, Hidden Layer {}".format(activation, size ))
            precisionRecallDriver( mlp, x_test, y_test,"KRKP/mlpPlots/PrecisionRecall-{}-{}-{}.png".format(activation, size[0], size[1] ), "Activation {}, Hidden Layer {}".format(activation, size ))
            y_score = mlp.predict_proba( x_test )
            roc_score = roc_auc_score(y_test, y_score[:,1])
            fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
            if i == 0:
                plt.clf()
                plt.xlabel('False Positive Ratio')
                plt.ylabel('True Positive Ratio')
                plt.title("mlp ROC Curve: activation = {}".format(activation))
            plt.plot( fpr, tpr, color=colors[i] )

            truePosRate = tp/(tp+fn)
            falsePosRate = fp/(fp+tn)

            t1 = time.time()
            runTime = t1 - t0
            myTuple = [size, activation, tp, fn, fp, tn, truePosRate, falsePosRate, pr_score, scores.mean(), scores.std(), roc_score, error, runTime, "mlp"]
            g_list.append(myTuple)
            i = i + 1
        plt.legend(["Hidden layer:(2, 2)", "Hidden layer:(5, 2)", "Hidden layer:(10, 2)", "Hidden layer:(2, 5)", "Hidden layer:(2, 10)", "Hidden layer:(10, 10)", "Hidden layer:(50, 2)"], loc ="lower right")
        plt.savefig("KRKP/mlpPlots/ROC-A-{}.png".format( activation))


if __name__ == '__main__':
    # Decision Trees, Bagging, Boosting, Logistic Regression, KNN
    # process data into scikit usable form
    init()
    t0 = time.time()
    print("0% ~ 270s to go")
    x_train, x_test, y_train, y_test = processKRKP()
    decision_tree_driverKRvKP( x_train, x_test, y_train, y_test )
    print("17%")
    bagging_dtree_driverKRVKP( x_train, x_test, y_train, y_test )
    print("34%")
    boosting_dtree_driverKRVKP( x_train, x_test, y_train, y_test )
    print("51%")
    knn_driver_KRVKP( x_train, x_test, y_train, y_test )
    print("68%")
    logReg_driver_KRVKP( x_train, x_test, y_train, y_test )
    print("85%")
    mlp_driver_KRVKP( x_train, x_test, y_train, y_test )
    print("100%")
    t1 = time.time()
    runTime = t1 - t0
    print("runTime = {}".format(runTime))
    for myTuple in g_list:
        print(myTuple)
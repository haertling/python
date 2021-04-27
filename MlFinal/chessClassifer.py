'''
    File name: chessClassifier.py
    Author: Steven Haertling
    Date created: 4/24/2021
    Date last modified: 4/24/2021
    Python Version: 3.8.6
'''


#imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def transformKRKP( S ):
    matrix = np.empty_like(S[:,0])
    for i in range(len(S[0])):
        enc = preprocessing.LabelEncoder()
        col_vector = enc.fit_transform(S[:,i])
        matrix = np.c_[matrix, col_vector]
    return matrix[:, 1:37], matrix[:, 37]

def decision_tree_driver():
    S = np.genfromtxt("data/kr-vs-kp.data", missing_values=0, skip_header=0, delimiter=',', dtype=str)
    x, y = transformKRKP( S )
    testSize = round(len(y)*.7)
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=testSize, random_state=4 )
    
    depths = [1,3,5,10,20,30,40,50]
    features = [1,2,3,4,5,10,15]
    for depth in depths:    
        for feature in features:
            d_tree = tree.DecisionTreeClassifier(max_depth=depth,max_features=feature)
            decision_tree = d_tree.fit(x_train, y_train)
            prediction = decision_tree.predict(x_test)
            # confusion = confusion_matrix( y_test, prediction )
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
            error = accuracy_score( y_test, prediction )
            print("max depth = {}, feature splits = {}".format(depth, feature))
            print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}".format(tp,fn,fp,tn))
            print("accuracy = {}".format(error))

if __name__ == '__main__':
    decision_tree_driver()
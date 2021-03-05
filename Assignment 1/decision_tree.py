# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    values = np.unique ( x )
    size = len( values )
    dictionary = {values[k]:[] for k in range(size) }
    # dictionary = {}
    for i in range( len(x) ):
        for j in range( size ):
            if( x[i] == values[j] ):
                dictionary[values[j]].append(i)
                break

    return dictionary


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    values, counts = np.unique (  y, return_counts=True )
    total = np.sum( counts )
    entropy = 0
    for i in range(len(counts)):
        entropy += (-counts[i]/total)*np.log2(counts[i]/total)
    return entropy

def value_entropy( x, y ):
    Ydict = partition(y)
    Xdict = partition(x)
    Ypositive = Ydict[1]
    values, counts = np.unique( x, return_counts=True )
    total_count = np.sum(counts)
    entropy = []
    for i in range( len(Xdict) ):
        # numerator = length of the values[i] array that overlaps with y==1 at those positions
        # denomenator = counts[i] since we are in the subset
        prob = len( list( set(Xdict[values[i]]) & set(Ypositive) ) ) / counts[i]
        entropy.append(( counts[i] / total_count ) * probability_entropy(prob))
        # print("value {} = {}".format( i, entropy[i]))
    return entropy

def probability_entropy(x):
    if(x == 1):
        return 1
    if(x == 0):
        return 0
    entropy = - x * np.log2( x ) - ( 1 - x ) * np.log2( 1 - x )
    return entropy

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    totalSet = entropy(y)
    weightedSet = np.sum(value_entropy( x, y ))
    info = totalSet - weightedSet
    return info

def pairs( matrix ):
    rows, cols = matrix.shape
    attribute_value_pairs = []
    for i in range( 1, cols ):
        values = np.unique ( matrix[:, i] )
        for j in range( len( values ) ):
            attribute_value_pairs.append((i, values[j]))
    return attribute_value_pairs

def find_value(x , pos):

    values = np.unique( x )
    value = values[pos]
    return value

def split_set( x, y, value ,feature, bool):
    Ydict = partition(y)
    Xdict = partition( x[:, feature] )
    values, counts = np.unique( x[:, feature], return_counts=True )

    if( bool == False ):
        yes_values = Xdict[value]
    if( bool == True ):
        no_values = []
        for i in range( len( values ) ):
            if( values[i] != value ):
                no_values += Xdict[values[i]]
        # print("no values = {}".format(no_values))
        no_values.sort()
        # print("no values = {}".format(no_values))
    copy_y = y
    copy_x = x
    if( bool == False ):
        for pos in range( len( yes_values ) ):
            copy_y = np.delete( copy_y , yes_values[pos] - pos, 0)
            copy_x = np.delete( copy_x , yes_values[pos] - pos, 0)
        sub_y = copy_y
        sub_x = copy_x
        # print("left y\n{}".format(sub_y))
        # print("left x\n{}".format(sub_x))
    if( bool == True ):
        for pos in range( len( no_values ) ):
            # print("position = {}\nindex={}".format(no_values[pos], pos))
            copy_y = np.delete( copy_y , no_values[pos] - pos, 0)
            copy_x = np.delete( copy_x , no_values[pos] - pos, 0)
        sub_y = copy_y
        sub_x = copy_x
        # print("right y\n{}".format(sub_y))
        # print("right x\n{}".format(sub_x))
    return sub_x, sub_y

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    values, counts = np.unique( y, return_counts=True )
    max_place = np.argmax(counts)
    majority_label = values[max_place]
    # print("y majority: {}".format(majority_label))
    # 1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
    if len( values ) <= 1:
        return values[0]
    # 2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common value of y (majority label)
    elif len(attribute_value_pairs) == 0:
        return majority_label
    # 3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    elif depth == max_depth:
        return majority_label
    # start the tree
    else:
        # select best feature to split on
        rows, cols = x.shape
        attribute_list = []
        for i in range( cols ):
            attribute_list.append(mutual_information( x[:, i], y ))
            # print("feature {} gain: {}".format( i+1, attribute_list[i] ))
        best_attribute = np.argmax(attribute_list)
        # print("best: {}".format(best_attribute+1))
        value_of_best_features = value_entropy( x[:, best_attribute], y)
        # print("{}".format(value_of_best_features))
        best_value_location = np.argmin(value_of_best_features)

        value_pair = find_value( x[:, best_attribute] , best_value_location )
        # print("value of pair = {}".format(value_pair))
        tree = {}
        # print("attribute_value_pairs = \n{}".format(attribute_value_pairs))
        # print("trying to remove pair( {}, {} )".format( best_attribute+1, value_pair ))
        try:
            attribute_value_pairs.remove((best_attribute+1,value_pair))
        except:
            pass
        boolean = [ False, True ]
        for i in range( 2 ):
            # if( i == 1):
            #     print("starting right tree")
            x_subset, y_subset = split_set( x, y, value_pair, best_attribute, boolean[i] )
            values, counts = np.unique( y_subset, return_counts=True )
            if( len(values)  == 0 ):
                return majority_label
            tree[best_attribute+1, value_pair, boolean[i]] = {}
            # print("tree = {}".format(tree))
            # print(tree)
            subtree = id3(x_subset, y_subset, attribute_value_pairs, depth+1, max_depth)
            tree[best_attribute+1, value_pair, boolean[i]] = subtree
        # print(tree)
    return tree

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # print(tree.keys())
    keys = list( tree.keys())
    attribute = keys[0][0]
    value = keys[0][1]
    # print("attribute = {}\n".format(attribute))
    if( x[attribute-1] == value ):# true
        result = tree[keys[1]]
    if( x[attribute-1] != value ):# false
        result = tree[keys[0]]

    if type( result ) is dict:
        return predict_example( x, result )
    else:
        return result

def getNodeCount( tree ):

    keys = list(tree.keys())
    # print(keys)
    if type( tree ) is dict:
        nodes = 1
        if type( tree[keys[0]] ) is dict:
            nodes += getNodeCount(tree[keys[0]])

        if type( tree[keys[1]] ) is dict:
            nodes += getNodeCount(tree[keys[1]])
    return nodes


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    # loop through the sum
    sum = 0
    n = len(y_true)
    for x in range( n ):
        if y_true[x] != y_pred[x]:
            sum+=1
    return ( (1/n) * sum )

def confusion_matrix( y_true, y_pred ):
    n = len(y_true)
    t_pos = 0
    f_pos = 0
    f_neg = 0
    t_neg = 0
    for x in range( n ):
        if y_true[x] == y_pred[x]:
            if y_pred[x] == 0:
                t_neg += 1
            if y_pred[x] == 1:
                t_pos += 1
        if y_true[x] != y_pred[x]:
            if y_pred[x] == 0:
                f_neg += 1
            if y_pred[x] == 1:
                f_pos += 1
    print("true positive  = {} false negative = {}\nfalse positive = {}  true negative = {}\n".format(t_pos,f_neg,f_pos,t_neg))
    return

def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


if __name__ == '__main__':
    # Load the training data
    learning_curves = 0
    Weak_Learners   = 0
    scikit_learn    = 0
    other_data_sets = 1
    if( learning_curves == 1):
        for j in range( 1, 4 ):
            print("Starting monks-{}".format(j))
            M = np.genfromtxt("data/monks-{}.train".format(j), missing_values=0, skip_header=0, delimiter=',', dtype=int)
            ytrn = M[:, 0]
            xtrn = M[:, 1:]
            attribute_value_pairs = pairs(M)

            # Load the test data
            M = np.genfromtxt("data/monks-{}.test".format(j), missing_values=0, skip_header=0, delimiter=',', dtype=int)
            ytst = M[:, 0]
            Xtst = M[:, 1:]

            for i in range( 1, 11):
                # Learn a decision tree of depth 3
                decision_tree = id3(xtrn, ytrn, attribute_value_pairs, max_depth=i)
                # nodes = getNodeCount(decision_tree)
                # visualize(decision_tree)

                # Compute the test error
                y_pred = [predict_example(x, decision_tree) for x in Xtst]
                tst_err = compute_error(ytst, y_pred)
                # confusion_matrix( ytst, y_pred )
                print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
                # accuracy value
                # print('{0:4.2f}%'.format(100-(tst_err * 100)))

                # number of nodes in tree
                # print('{}'.format(nodes))
    if( Weak_Learners == 1 ):
        M = np.genfromtxt("data/monks-1.train", missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        xtrn = M[:, 1:]

        attribute_value_pairs = pairs(M)

        # Load the test data
        M = np.genfromtxt("data/monks-1.test", missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        for i in range( 1, 3):
            # Learn a decision tree of depth 3
            decision_tree = id3(xtrn, ytrn, attribute_value_pairs, max_depth=i)
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            print("monks-1, depth = {}".format(i))
            confusion_matrix( ytst, y_pred )

    if (scikit_learn == 1):
        import graphviz
        from sklearn import tree

        # Load the training data
        M = np.genfromtxt("data/monks-1.train", missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        xtrn = M[:, 1:]
        # Load the test data
        M = np.genfromtxt("data/monks-1.test", missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]
        # learn a tree
        d_tree = tree.DecisionTreeClassifier()
        decision_tree = d_tree.fit(xtrn, ytrn)
        # visualize the learning tree
        dot_data = tree.export_graphviz(decision_tree, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("monks-1-learning")

        # confusion matrix for test
        prediction = decision_tree.predict(Xtst)
        confusion_matrix( ytst, prediction )
        tst_err = compute_error( ytst, prediction )
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    if( other_data_sets == 1 ):
        M = np.genfromtxt("data/balance-scale.train", missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        xtrn = M[:, 1:]
        # print(ytrn)
        # print(xtrn)
        attribute_value_pairs = pairs(M)
        # print(attribute_value_pairs)
        M = np.genfromtxt("data/balance-scale.test", missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        for i in range( 1, 3):
            # Learn a decision tree of depth 3
            decision_tree = id3(xtrn, ytrn, attribute_value_pairs, max_depth=i)
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            print("balance-scale, depth = {}".format(i))
            confusion_matrix( ytst, y_pred )
        # learn a tree
        d_tree = tree.DecisionTreeClassifier()
        decision_tree = d_tree.fit(xtrn, ytrn)
        # visualize the learning tree
        dot_data = tree.export_graphviz(decision_tree, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("balance-scale-training")
        # confusion matrix for test
        prediction = decision_tree.predict(Xtst)
        confusion_matrix( ytst, prediction )
        tst_err = compute_error( ytst, prediction )
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

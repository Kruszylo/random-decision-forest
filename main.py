#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:04:34 2018

@author: kruszylo
"""
import sys
import numpy as np
from numpy import genfromtxt
from Tree import Tree
import copy
import random 

problem_data = None 

# A utility function to find deepest leaf node. 
# lvl:  level of current node. 
# maxlvl: pointer to the deepest left leaf node found so far 
# isLeaf: A bool indicate that this node is child 
# of its parent 
# leafPtr: Pointer to the leaf 
# parentPtr: Pointer to the parent of the leaf
def deepestLeaf(root, parent, lvl, maxlvl): 
      
    # Base Case 
    if root is None: 
        return
  
    # Update result if this node is leaf and its  
    # level is more than the max level of the current result 
    # if(isLeaf is True): 
    if (root.left == None and root.right == None): 
        if lvl > maxlvl[0] :  
            deepestLeaf.leafPtr = root  
            deepestLeaf.leafParentPtr = parent
            maxlvl[0] = lvl  
            return
  
    # Recur for left and right subtrees 
    deepestLeaf(root.left, root, lvl+1, maxlvl) 
    deepestLeaf(root.right, root, lvl+1, maxlvl) 
  
# A wrapper for left and right subtree 
def deepestLeafParent(root): 
    maxlvl = [0] 
    deepestLeaf.leafPtr = None
    deepestLeaf.leafParentPtr = None
    deepestLeaf(root, None, 0, maxlvl, False) 
    return deepestLeaf.leafParentPtr 
    
def create_dataset(train_proportion = 5, validation_proportion = 1, test_proportion = 1):
    data = genfromtxt('data/winequality-white.csv', delimiter=';', names=True)
    
    total_slices = int(train_proportion)+int(validation_proportion) + int(test_proportion)
    
    np.random.shuffle(data)
    train_len = int((len(data)/total_slices)*int(train_proportion))
    validation_len = int((len(data)/total_slices)*int(validation_proportion))
    
    train_set, validation_set, test_set = data[:train_len], data[train_len:train_len + validation_len], data[train_len + validation_len:]
    return train_set, validation_set, test_set

def grow_forest(dataset, K):
    forest = []
    all_Xt_labels = dataset.dtype.names[:-1]
    labels_num = int(np.sqrt(len(all_Xt_labels)))
    for i in range(K):
        Xt_labels = random.sample(all_Xt_labels, labels_num)
        print(f'Xt_labels {Xt_labels}')
        tree = create_tree(dataset, Xt_labels)
        print(f'Tree {i} created')
#        key = input('continue?')
#        if key == 'n':
#            sys.exit(2)
        forest.append(tree)
    return forest

def evaluete_forest(forest, dataset):
    success = 0
    qualities = []
    for row in dataset:
        forest_quality_votes = [] 
        for tree in forest:
            forest_quality_votes.append(tree.evaluete(row))
        best_voted_quality = max(set(forest_quality_votes), key=forest_quality_votes.count)
        if best_voted_quality == row['quality']:
            success+=1
        qualities.append(best_voted_quality)
    return success/len(dataset), qualities

def create_tree(dataset, Xt_labels):
    key = input('continue?')
    if key == 'n':
        sys.exit(2)
    elif key == 'p':
        print(dataset)
        global problem_data
        problem_data = dataset
    if len(set(dataset['quality'])) == 1:
        leaf = Tree()
        leaf.quality = dataset['quality'][0]
        leaf.num_of_samples = len(dataset)
        return leaf
    if len(dataset) == 0:
        return None
    node = Tree()
    Xt, Xt_dict,min_xt_dict = get_min_xt(dataset,Xt_labels)
    #if all information is same BUT quality classes are different
    if len(Xt_dict) > 1 and len(set(Xt_dict.values())) == 1:
        leaf = Tree()
        leaf.quality = int(sum(dataset['quality'])/len(dataset))
        leaf.num_of_samples = len(dataset)
        return leaf
    node.xt = min_xt_dict[Xt]
    node.Xt = Xt
    #print(f'node.Xt = {node.Xt}, node.xt = {node.xt}')
    less_subdataset = list()
    greater_subdataset = list()
    for row in dataset:
            if row[node.Xt] <= node.xt:
                less_subdataset.append(row)
            else:
                greater_subdataset.append(row)
    print(f'Xt = {Xt}, xt = {min_xt_dict[Xt]} -- going left, len(dataset)={len(dataset)}, len(less)={len(less_subdataset)}, len(greater)={len(greater_subdataset)}')
    node.left = create_tree(np.array(less_subdataset), Xt_labels)
    #print(f'going right, len(dataset)={len(dataset)}, len(less)={len(less_subdataset)}, len(greater)={len(greater_subdataset)}')
    node.right = create_tree(np.array(greater_subdataset), Xt_labels)
    return node

def evaluete_tree(root, dataset):
    head = root
    success = 0
    qualities = []
    for row in dataset:
        tree = head
        quality = tree.evaluete(row)
        if quality == row['quality']:
            success+=1
        qualities.append(quality)
    return success/len(dataset), qualities

def reduce_node(parent, child):
    parent.Xt = None
    parent.xt = None
    parent.quality = child.quality
    parent.num_of_samples = child.num_of_samples
    parent.left = None
    parent.right = None
        
def prune_tree(root, validation_set, epsilon = 0):
    # deepest_leaf_parent = deepestLeafParent(root)
    original_root = copy.deepcopy(root)
    leaves_prumed = 0
    #cut tree
    leaf_parents_list = set()
    alerady_tested_nodes = list()
    root.get_leaves_parents(root, None, leaf_parents_list)
    while len(leaf_parents_list) > 0:
        print(f'Leaves which are were not pruned yet: {len(leaf_parents_list)}')
        acc1, _ = evaluete_tree(root, validation_set)
        deepest_leaf_parent = (list(leaf_parents_list)[0])
        leaf_copy = copy.deepcopy(list(leaf_parents_list)[0])
        if deepest_leaf_parent.left != None and deepest_leaf_parent.left.num_of_samples != None:
            if deepest_leaf_parent.right != None and deepest_leaf_parent.right.num_of_samples != None:
                if deepest_leaf_parent.left.num_of_samples > deepest_leaf_parent.right.num_of_samples:
                    reduce_node(deepest_leaf_parent, deepest_leaf_parent.left)
                else:
                    reduce_node(deepest_leaf_parent,deepest_leaf_parent.right)
            else:
                reduce_node(deepest_leaf_parent,deepest_leaf_parent.left)
        else:
            reduce_node(deepest_leaf_parent, deepest_leaf_parent.right)
        acc2, _ = evaluete_tree(root, validation_set)
        if acc2 >= acc1 - epsilon:
            if deepest_leaf_parent.Xt != None:
                leaf_parents_list.remove(deepest_leaf_parent)
            else:
                leaf_parents_list = list(leaf_parents_list)
                for i,leaf in enumerate(leaf_parents_list):
                    if leaf.quality == deepest_leaf_parent.quality and leaf.num_of_samples == deepest_leaf_parent.num_of_samples:
                        
                        del leaf_parents_list[i]
                        leaf_parents_list = set(leaf_parents_list)
                        break
            print('PRUMED ONE LEAF')
            leaves_prumed+=1
            
        else:
            alerady_tested_nodes.append(leaf_copy)
            root = copy.deepcopy(original_root)
            leaf_parents_list = set()
            root.get_leaves_parents(root, None, leaf_parents_list)
            for alerady_tested_node in alerady_tested_nodes:
                if alerady_tested_node in leaf_parents_list:
                     leaf_parents_list.remove(alerady_tested_node)
        original_root = copy.deepcopy(root)
    print(f'POSTPRUMING FINISHED! IN TOTAL PRUMED {leaves_prumed} LEAVES')
    return original_root

def entropy(X, dataset):
    prob = get_probabilities(X, dataset)
    entr = 0            
    for _, probability in prob.items():
        entr += probability * np.log2(probability)
    return -entr

def get_probabilities(X, dataset, Y='quality'):
    prob = dict()
    for row in dataset:
        if row[X] in prob:
            prob[row[X]] +=1
        else:
            prob[row[X]] = 1
    for x in prob.keys():
        prob[x] /= len(dataset)
    return prob

def get_probabilities_2d(X, dataset, Y='quality'):
    total = len(dataset)
    if total == 0:
        return dict()
    #init cond prob table
    #prob_2d = np.zeros((len(set(dataset[Y])),len(set(dataset[X]))))
    prob_2d =  dict()
    for row in dataset:
        if row[Y] in prob_2d:
            prob_2d[row[Y]] += 1
        else:
            prob_2d[row[Y]] = 1
    for y in prob_2d.keys():
            prob_2d[y] = prob_2d[y]/total
            
    return prob_2d

def entropy_2d(prob_2d):
    entr_2d = 0
    for y in prob_2d.keys():
            entr_2d += prob_2d[y] * np.log2(prob_2d[y])
            
    return -entr_2d

def get_min_xt_in_Xt(Xt,dataset):
    xt_values = set(dataset[Xt])
    X_prob = get_probabilities(Xt, dataset)
    xt_avr_classif_uncertaintly = dict()
    for xt in xt_values:
        #get rows where x less than xt
        less_subdataset = list()
        less_comulated_prob = 0
        greater_subdataset = list()
        greater_comulated_prob = 0
        for row in dataset:
            if row[Xt] <= xt:
                less_subdataset.append(row)
            else:
                greater_subdataset.append(row)
        less_or_eq_entropy = entropy_2d(get_probabilities_2d(Xt, np.array(less_subdataset)) ) #- entropy(Xt, np.array(less_subdataset))
        greater_than_entropy = entropy_2d(get_probabilities_2d(Xt, np.array(greater_subdataset)) ) #- entropy(Xt, np.array(greater_subdataset))
        
        for x in X_prob:
            if x <= xt:
                less_comulated_prob += X_prob[x]
            else:
                greater_comulated_prob += X_prob[x]
        xt_avr_classif_uncertaintly[xt] = less_comulated_prob * less_or_eq_entropy + greater_comulated_prob * greater_than_entropy
        
    return min(xt_avr_classif_uncertaintly, key=xt_avr_classif_uncertaintly.get), xt_avr_classif_uncertaintly

def get_min_xt(dataset, Xt_labels):
    Xt_min, Xt_min_values = dict(), dict()
    for Xt in Xt_labels:
        min_Xt_ind, min_Xt_dict = get_min_xt_in_Xt(Xt, dataset)
        Xt_min[Xt], Xt_min_values[Xt] = min_Xt_dict[min_Xt_ind], min_Xt_ind
    return min(Xt_min, key=Xt_min.get), Xt_min, Xt_min_values

def main(argv):
    if len(argv)>=3:
        train_set, validation_set, test_set = create_dataset(argv[0],  argv[1], argv[2])
        Xt_labels = train_set.dtype.names[:-1]
        root = create_tree(train_set, Xt_labels)
        print('tree created')
        test_acc, _ = evaluete_tree(root, test_set)
        print(f'Tree acc on test set: {test_acc}')
    else:
        print('main.py <train_proportion> <validation_proportion> <test_proportion>')
#        sys.exit(2)
if __name__ =="__main__":
    main(sys.argv[1:])
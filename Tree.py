#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:14:56 2018

@author: kruszylo
"""

class Tree(object):
    def __init__(self, left = None, right = None,quality = None, num_of_samples = None):
        self.quality = quality
        self.num_of_samples = num_of_samples
        self.left = left
        self.right = right
        self.Xt = None
        self.xt = None
        
    def evaluete(self, row):
        while self.Xt != None:
            if row[self.Xt] <= self.xt:
                self = self.left
            else:
                self = self.right
        return self.quality
    
    def get_leaves_parents(self, root, parent, parents):
        if root is None: 
            return
    
        if (root.left == None and root.right == None):
            parents.add(parent)
            return
        self.get_leaves_parents(root.left,root,parents)
        self.get_leaves_parents(root.right,root,parents)
        
    def __hash__(self):
        return hash((self.Xt, self.xt))
  
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.Xt == other.Xt and self.xt == other.xt

    def __lt__(self, other):
        return self.xt < other.xt
        
    
    
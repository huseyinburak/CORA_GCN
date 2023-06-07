#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 07:06:21 2023

@author: hbonen
"""

import networkx as nx
from networkx import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

import os

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

from bokeh.palettes import Spectral

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Initialize the graph
G = nx.Graph(name='G')

all_data = []
all_edges = []

for root,dirs,files in os.walk('./cora'):
    for file in files:
        if '.content' in file:
            with open(os.path.join(root,file),'r') as f:
                all_data.extend(f.read().splitlines())
        elif 'cites' in file:
            with open(os.path.join(root,file),'r') as f:
                all_edges.extend(f.read().splitlines())

                
#Shuffle the data because the raw data is ordered based on the label
random_state = 77
all_data = shuffle(all_data,random_state=random_state)

#parse the data
labels = []
nodes = []
X = []

for i,data in enumerate(all_data):
    elements = data.split('\t')
    labels.append(elements[-1])
    # X.append(elements[1:-1])
    nodes.append(int(elements[0]))


#parse the edge
edge_list=[]
for edge in all_edges:
    e = edge.split('\t')
    edge_list.append((int(e[0]),int(e[1])))

# print('\nNumber of nodes (N): ', N)
# print('\nNumber of features (F) of each node: ', F)
sL = set(labels)
print('\nCategories: ', sL)
nsL = len(sL)

N=len(nodes)
for j in range(N-1):
    G.add_node(nodes[j], name=nodes[j])

G.add_edges_from(edge_list)

plt.figure(figsize=(80, 80))
pos = nx.kamada_kawai_layout(G)
# pos = nx.circular_layout(G)
options = {
    'node_color': 'red',  # first nsL colors from the Spectral palette
    'node_size': 3,
    'width': 0.5,
    'arrowstyle': '-',
    'arrowsize': 0.6,
}

nx.draw_networkx(G, pos=pos, with_labels = False, arrows=True, **options)

ax = plt.gca()
ax.collections[0].set_edgecolor("#000000")

plt.show()

A = np.array(nx.attr_matrix(G)[0])
X = np.array(nx.attr_matrix(G)[1])
X = np.expand_dims(X,axis=1)

print('Shape of A: ', A.shape)
print('\nShape of X: ', X.shape)

AX = np.dot(A,X)

G_self_loops = G.copy()

self_loops = []
for r in range(N-1):
    self_loops.append((nodes[r],nodes[r]))

G_self_loops.add_edges_from(self_loops)

A_hat = np.array(nx.attr_matrix(G_self_loops)[0])

AX = np.dot(A_hat, X)

Deg_Mat = G_self_loops.degree()

D = np.diag([deg for (n,deg) in list(Deg_Mat)])

D_inv = np.linalg.inv(D)

DAX = np.dot(D_inv,AX)

D_half_norm = fractional_matrix_power(D, -0.5)
DADX = D_half_norm.dot(A_hat).dot(D_half_norm).dot(X)

np.random.seed(77777)
n_h = 14 #number of neurons in the hidden layer
n_y = 7 #number of neurons in the output layer
W0 = np.random.randn(X.shape[1],n_h) * 0.01
W1 = np.random.randn(n_h,n_y) * 0.01

#Implement ReLu as activation function
def relu(x):
    return np.maximum(0,x)

def gcn(A,H,W):
    I = np.identity(A.shape[0]) #create Identity Matrix of A
    A_hat = A + I #add self-loop to A
    D = np.diag(np.sum(A_hat, axis=0)) #create Degree Matrix of A
    D_half_norm = fractional_matrix_power(D, -0.5) #calculate D to the power of -0.5
    eq = D_half_norm.dot(A_hat).dot(D_half_norm).dot(H).dot(W)
    return relu(eq)

H1 = gcn(A,X,W0)
H2 = gcn(A,H1,W1)
print('Features Representation from GCN output:\n', H2)

def plot_features(H2):
    x = H2[:,0]
    y = H2[:,4]

    size = 5
    plt.figure()
    plt.scatter(x,y,size)
    plt.xlim([np.min(x)*0.9, np.max(x)*1.1])
    plt.ylim([-1, 1])
    plt.xlabel('Feature Representation Dimension 0')
    plt.ylabel('Feature Representation Dimension 1')
    plt.title('Feature Representation')

    for q,row in enumerate(H2):
        str = "{}".format(q)
        plt.annotate(str, (row[0],row[1]),fontsize=18, fontweight='bold')
    plt.show()
    
plot_features(H2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from scipy.sparse import csr_matrix
from data_helper import train_tiedAE
from sklearn import preprocessing

def load_data(data):
	if data == "Alibaba-s":
		data = np.loadtxt('data/Alibaba_s/alibaba_s.txt', dtype=int)
		return data, None
	elif data == "Alibaba":
		data = np.loadtxt('data/Alibaba/alibaba_edges.txt', dtype=int)
		feats_u = np.loadtxt('data/Alibaba/alibaba_featu.txt', dtype=float)
		feats_v = np.loadtxt('data/Alibaba/alibaba_featv.txt', dtype=float)
		feats = np.concatenate((feats_u,feats_v),axis=0)
		return data, feats
	elif data == "DTI":
		data = np.loadtxt('data/DTI/DTI.txt', dtype=int)
		return data, None
	elif data == "Amazon":
		data = np.loadtxt('data/Amazon/amazon_edges.txt', dtype=int)
		feats_u = np.loadtxt('data/Amazon/amazon_featu.txt', dtype=float)
		feats_v = np.loadtxt('data/Amazon/amazon_featv.txt', dtype=float)
		feats = np.concatenate((feats_u,feats_v),axis=0)
		return data, feats

def load_attributes(data, feats):
	# print(feats.shape)
	num_btype = len(np.unique(data[:,2]))
	btypes = [str(i) for i in range(num_btype)]
	# btypes = []
	btypes.append('base')
	# print(btypes)
	init_feats = dict()
	for btype in btypes:
		init_feats[btype] = feats #preprocessing.normalize(feats)
	return init_feats

def split_train_test(data, feats, flag, ratio):
	n_samples = data.shape[0]
	n_test = int(n_samples*ratio)
	ridx = np.random.choice(n_samples, n_test, replace=False)
	test = data[ridx]
	train = np.delete(data, ridx, axis=0)
	print(test.shape,train.shape)
	train_nodes_u = [i for i in list(set(train[:,0]))]
	train_nodes_i = [i for i in list(set(train[:,1]))]
	# print(len(train_nodes_u),len(train_nodes_i))
	train_ui = train_nodes_u
	for i in train_nodes_i:
		train_ui.append(i)
	# if feats != None:
	# 	feats_train = np.array([feats[i] for i in train_ui])
	# 	print(feats.shape, feats_train.shape)
	u_train,i_train = np.unique(train[:,0]),np.unique(train[:,1])
	f_test = []
	for line in test:
		if line[0] in u_train and line[1] in i_train:
			f_test.append(line)
	f_test = np.array(f_test)
	# print("### The train contains %d edges(%d drugs and %d targets), and test contains %d edges(%d drugs and %d targets)." 
	# 	% (train.shape[0],len(u_train),len(i_train),f_test.shape[0],len(np.unique(f_test[:,0])),len(np.unique(f_test[:,1]))))
	idx_u_map = {j:i for i,j in enumerate(np.unique(train[:,0]))}
	idx_i_map = {j:len(np.unique(train[:,0]))+i for i,j in enumerate(np.unique(train[:,1]))}
	new_train,new_test = [],[]
	for line in train:
		tmp = []
		tmp.append(idx_u_map[line[0]])
		tmp.append(idx_i_map[line[1]])
		tmp.append(line[2])
		# tmp.append(line[3]) # label
		new_train.append(tmp)
	for line in f_test:
		tmp = []
		tmp.append(idx_u_map[line[0]])
		tmp.append(idx_i_map[line[1]])
		tmp.append(line[2])
		# tmp.append(line[3])
		new_test.append(tmp)
	new_train,new_test = np.array(new_train),np.array(new_test)
	u_train_new,i_train_new = np.unique(new_train[:,0]),np.unique(new_train[:,1])
	# u_test_new,i_test_new = np.unique(new_test[:,0]),np.unique(new_test[:,1])
	# print("### The train contains %d edges(%d drugs and %d targets), and test contains %d edges(%d drugs and %d targets)." 
		# % (new_train.shape[0],len(u_train_new),len(i_train_new),new_test.shape[0],len(u_test_new),len(i_test_new)))
	print(len(np.unique(new_train[:,0])),len(np.unique(new_train[:,1])))
	if flag == True:
		feats_train = np.array([feats[i] for i in train_ui])
		return new_train, new_test, feats_train
	elif flag == False:
		return new_train, new_test, None

def extract_hyedges_types(data, btype):
	new_data = []
	for line in data:
		if line[2] == btype:
			new_data.append(line)
	return np.array(new_data)

def construct_hierarchical_hypergraph(data, nodes):
	hygraphs = dict()
	btypes = list(set(data[:,2])) #[0,1,2,3,4]
	hygraph_base = construct_hypergraph(data, nodes)
	hygraphs['base'] = hygraph_base
	for btype in btypes:
		data_type = extract_hyedges_types(data, btype)
		hygraphs[str(btype)] = construct_hypergraph(data_type, nodes)
	return hygraphs

def construct_hypergraph(data, nodes):
	# construct Bigraph
	nodes_u = [i for i in list(set(data[:,0]))]
	nodes_i = [i for i in list(set(data[:,1]))]
	all_nodes_u,all_nodes_i = nodes['u'],nodes['i']
	print(len(nodes_u),len(nodes_i),len(all_nodes_u),len(all_nodes_i))
	Bigraph = nx.Graph()
	for line in data:
		node_u,node_i = line[0],line[1]
		Bigraph.add_node(node_u, bipartite=0)
		Bigraph.add_node(node_i, bipartite=1)
		Bigraph.add_edge(node_u, node_i, btype=line[2])
	# construct Hygraph
	Hygraph = dict()
	n_neigs_u = 0
	n_neigs_i = 0
	for u in all_nodes_u:
		if u in nodes_u:
			neighbors = Bigraph.edges(u)
			neigs_u = [i for u,i in neighbors]
			Hygraph[u] = neigs_u
			n_neigs_u += len(neigs_u)
		else:
			Hygraph[u] = []
	for i in all_nodes_i:
		if i in nodes_i:
			neighbors = Bigraph.edges(i)
			neigs_i = [u for i,u in neighbors]
			Hygraph[i] = neigs_i
			n_neigs_i += len(neigs_i)
		else:
			Hygraph[i] = []
	print('### The number of hyper-edges: %d' %(len(nodes_u)+len(nodes_i)))
	print('### The average nodes in each hyper-edge: %0.2f (%0.2f for drugs and %0.2f for targets)' 
		% ((n_neigs_u+n_neigs_i)/(len(nodes_u)+len(nodes_i)),n_neigs_u/len(nodes_u),(n_neigs_i/len(nodes_i))))
	return Hygraph

def generate_adj(data,nodes,btype):
	N_u = nodes['n_u']#len(np.unique(data[:,0]))
	N_i = nodes['n_i']#len(np.unique(data[:,1]))
	N = N_u + N_i
	adj = np.zeros((N, N), dtype=int)
	if btype == 'base':
		for line in data:
			adj[line[0],line[1]] = 1
	else:
		for line in data:
			if line[2] == int(btype):
				adj[line[0],line[1]] = 1
	# print(np.sum(adj == 1))
	return csr_matrix(adj).astype('float32')

def initialize_features(args, data, nodes, dim=32):
	# Encoder Based Approach
	print('### Generating initial features by Encoder-Based-Approach...')
	num_btype = len(np.unique(data[:,2]))
	btypes = [str(i) for i in range(num_btype)]
	# btypes = []
	btypes.append('base')
	initial_feats = dict()
	for btype in btypes:
		A = generate_adj(data,nodes,btype).todense()
		initial_feat = train_tiedAE(A,dim=args.dim_f,lr=args.lr_eba,weight_decay=args.weight_decay_eba,n_epochs=args.epoch_eba)
		initial_feats[btype] = preprocessing.normalize(initial_feat)
	return initial_feats

def generate_incidence_matrix_multiple(hygraphs):
	Hs = dict()
	btypes = hygraphs.keys()
	n_smp = len(hygraphs['base'])
	for btype in btypes:
		H = generate_incidence_matrix(hygraphs[btype],n_smp)
		Hs[btype] = H
	return Hs

def generate_incidence_matrix(hyedges, n_smp):
	H = np.zeros((n_smp,n_smp))
	for key,val in hyedges.items():
		for v in val:
			H[v,key] = 1
	return H

def generate_negative_samples(pos_edges, hygraph, num_neg_samples):
	nodes_u = list(set([u for u,i in pos_edges]))
	nodes_i = list(set([i for u,i in pos_edges]))
	neg_edges = []
	for u in nodes_u:
		candidates = list(set(nodes_i) - set(hygraph[u]))
		neg_nodes = np.random.choice(candidates, num_neg_samples, replace=False)
		for neg in neg_nodes:
			tmp = [u, neg]
			neg_edges.append(tmp)
	for i in nodes_i:
		candidates = list(set(nodes_u) - set(hygraph[i]))
		neg_nodes = np.random.choice(candidates, num_neg_samples, replace=False)
		for neg in neg_nodes:
			tmp = [i, neg]
			neg_edges.append(tmp)
	neg_edges = np.array(neg_edges)
	np.random.shuffle(neg_edges)
	return neg_edges

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor,optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from time import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

### <!-- Dual Hypergraph Convolutional Network for learning hypergraphs --!> ###
class DualHGCN(nn.Module):
	def __init__(self, in_ch, n_hid, dty_nets, inter, intra, dropout=0.5):
		super(DualHGCN, self).__init__()
		self.dropout = dropout
		self.dty_nets = dty_nets
		self.dim_emb = n_hid[-1]
		self.inter = inter
		self.intra = intra
		self.HyperConv_1 = MultiHyperConv(in_ch, n_hid[0], self.dty_nets, self.inter, self.intra, self.dropout)
		self.HyperConv_2 = MultiHyperConv(n_hid[0], n_hid[1], self.dty_nets, self.inter, self.intra, self.dropout)
		self.Linear_u = nn.Linear(n_hid[-1]*len(dty_nets), n_hid[-1])
		self.Linear_i = nn.Linear(n_hid[-1]*len(dty_nets), n_hid[-1])
		print(dty_nets)

	def dropout_layer(self, Xu, Xi):
		out_xu,out_xi=dict(),dict()
		for dty in self.dty_nets:
			out_xu[dty] = F.dropout(Xu[dty], self.dropout)
			out_xi[dty] = F.dropout(Xi[dty], self.dropout)
		return out_xu,out_xi

	def forward(self, Xu, Xi, Gu, Gi, Hu, Hi):
		Xu_1,Xi_1 = self.HyperConv_1(Xu, Gu, Hu, Xi, Gi, Hi)
		Xu_1,Xi_1 = self.dropout_layer(Xu_1, Xi_1)
		Xu_2,Xi_2 = self.HyperConv_2(Xu_1, Gu, Hu, Xi_1, Gi, Hi)
		dty_nets = self.dty_nets-['base']

		all_xu = Xu_2['base']
		for dty in dty_nets:
			add_xu = Xu_2[dty]
			all_xu = torch.cat((all_xu, add_xu), 1)
		opt_xu = self.Linear_u(all_xu)

		all_xi = Xi_2['base']
		for dty in dty_nets:
			add_xi = Xi_2[dty]
			all_xi = torch.cat((all_xi, add_xi), 1)
		opt_xi = self.Linear_i(all_xi)

		opt_x = torch.cat((opt_xu, opt_xi), 0)
		return opt_x

class Embed_layer(nn.Module):
	def __init__(self, in_ft, out_ft, dty_nets):
		super(Embed_layer, self).__init__()
		self.weight = Parameter(torch.Tensor(in_ft, out_ft)).to(device)
		self.reset_parameters()
		self.dty_nets = dty_nets

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)

	def forward(self, X:torch.Tensor):
		X_ = dict()
		for dty in self.dty_nets:
			X_[dty] = X[dty].matmul(self.weight)
		return X_

class HyperConv(nn.Module):
	def __init__(self, in_ft, out_ft, inter=False, intra=True, bias=True):
		super(HyperConv, self).__init__()
		self.weight_u = Parameter(torch.Tensor(in_ft, out_ft)).to(device)
		self.weight_i = Parameter(torch.Tensor(in_ft, out_ft)).to(device)
		if bias:
			self.bias = Parameter(torch.Tensor(out_ft)).to(device)
		else:
			self.register_parameter(torch.Tensor(out_ft)).to(device)
		self.WB = Parameter(torch.Tensor(out_ft, out_ft)).to(device)
		self.reset_parameters()
		self.inter = inter
		self.intra = intra

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight_u.size(1))
		self.weight_u.data.uniform_(-stdv, stdv)
		self.weight_i.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)
		self.WB.data.uniform_(-stdv, stdv)

	def forward(self,Xu:torch.Tensor,Gu:torch.Tensor,Hu:torch.Tensor,Xi:torch.Tensor,Gi:torch.Tensor,Hi:torch.Tensor,B:torch.Tensor,_intra:torch.bool):
		Xu = Xu.matmul(self.weight_u)
		Xi = Xi.matmul(self.weight_i)
		X = Gu.matmul(Xu)
		if self.inter:
			HiT = torch.transpose(Hi,0,1)
			X = X + HiT.matmul(Xi)
		if self.intra and _intra:
			X = X + B.matmul(self.WB)
		if self.bias is not None:
			X = X + self.bias
		X = F.relu(X)
		return X

class MultiHyperConv(nn.Module):
	def __init__(self, in_ft, out_ft, dty_nets, inter, intra, dropout, bias=True):
		super(MultiHyperConv, self).__init__()
		self.dty_nets = dty_nets
		self.dropout = dropout
		self.HyperConv_U = dict()
		self.HyperConv_I = dict()
		for dty in self.dty_nets:
			self.HyperConv_U[dty] = HyperConv(in_ft, out_ft, inter=inter, intra=intra)
			self.HyperConv_I[dty] = HyperConv(in_ft, out_ft, inter=inter, intra=intra)

	def forward(self,Xu:torch.Tensor,Gu:torch.Tensor,Hu:torch.Tensor,Xi:torch.Tensor,Gi:torch.Tensor,Hi:torch.Tensor):
		self._dty_nets = self.dty_nets-['base']
		### HyperConv on U
		out_xu = dict()
		base_xu = self.HyperConv_U['base'](Xu['base'],Gu['base'],Hu['base'],Xi['base'],Gi['base'],Hi['base'],Xu,False)
		out_xu['base'] = base_xu
		for dty in self._dty_nets:
			add_xu = self.HyperConv_U[dty](Xu[dty],Gu[dty],Hu[dty],Xi[dty],Gi[dty],Hi[dty],base_xu,True)
			out_xu[dty] = add_xu

		### HyperConv on I
		out_xi = dict()
		base_xi = self.HyperConv_I['base'](Xi['base'],Gi['base'],Hi['base'],Xu['base'],Gu['base'],Hu['base'],Xi,False)
		out_xi['base'] = base_xi
		for dty in self._dty_nets:
			add_xi = self.HyperConv_I[dty](Xi[dty],Gi[dty],Hi[dty],Xu[dty],Gu[dty],Hu[dty],base_xi,True)
			out_xi[dty] = add_xi

		return out_xu,out_xi

def generate_G_from_H(args, H):
	H = np.array(H)
	n_edge = H.shape[1]
	W = np.ones(n_edge)
	DV = np.sum(H * W, axis=1)
	DE = np.sum(H, axis=0)
	DV += 1e-12
	DE += 1e-12
	invDE = np.mat(np.diag(np.power(DE, -1)))
	W = np.mat(np.diag(W))
	H = np.mat(H)
	HT = H.T
	if args.conv == "sym":
		DV2 = np.mat(np.diag(np.power(DV, -0.5)))
		G = DV2 * H * W * invDE * HT * DV2   #sym
	elif args.conv == "asym":
		DV1 = np.mat(np.diag(np.power(DV, -1)))
		G = DV1 * H * W * invDE * HT   #asym
	return G

def generate_Gs_from_Hs(args, Hs):
	Gs = dict()
	for key,val in Hs.items():
		Gs[key] = generate_G_from_H(args, val)
	return Gs

def split_Hs(Hs, num_u):
	Hs_u,Hs_i = dict(),dict()
	for key,val in Hs.items():
		Hs_u[key] = val[:num_u,num_u:]
		Hs_i[key] = val[num_u:,:num_u]
	return Hs_u,Hs_i

def embedding_loss(embeddings, positive_links, negtive_links, lamb):
	left_p = embeddings[positive_links[:, 0]]
	right_p = embeddings[positive_links[:, 1]]
	dots_p = torch.sum(torch.mul(left_p, right_p), dim=1)
	positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
	left_n = embeddings[negtive_links[:, 0]]
	right_n = embeddings[negtive_links[:, 1]]
	dots_n = torch.sum(torch.mul(left_n, right_n), dim=1)
	negtive_loss = torch.mean(-1.0 * torch.log(1.01 - torch.sigmoid(dots_n)))
	loss =  lamb*positive_loss + (1-lamb)*negtive_loss
	return loss

def train(args, model, X_u, X_i, samples, G_u, G_i, H_u, H_i):
	lr = args.lr
	weight_decay = args.weight_decay

	if args.optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif args.optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
	n_epoch = args.epoch
	feats_u, feats_i, target1, target2 = X_u, X_i, samples['pos_samples'], samples['neg_samples']

	for epoch in range(n_epoch):
		model.train()
		optimizer.zero_grad()
		embeds = model.forward(feats_u, feats_i, G_u, G_i, H_u, H_i)
		loss = embedding_loss(embeds, target1, target2, args.lamb)
		loss.backward()
		optimizer.step()
		if (epoch+1) % 100 == 0 or epoch == 0:
			print('The loss of %d-th epoch: %0.4f' % (epoch+1, loss))
	model.eval()
	outputs = model.forward(feats_u, feats_i, G_u, G_i, H_u, H_i)
	return outputs

def train_DualHGCN(args, X, Hs, samples, num_u):
	Xs_u,Xs_i = dict(),dict()
	for key,val in X.items():
		X_ = X[key]
		X_u,X_i = X_[:num_u,:],X_[num_u:,:]
		Xs_u[key] = Tensor(X_u).to(device)
		Xs_i[key] = Tensor(X_i).to(device)
	# n_sample = X.shape[0]
	in_ft = X['base'].shape[1]
	H_u,H_i = split_Hs(Hs,num_u)
	G_u = generate_Gs_from_Hs(args, H_u)
	G_i = generate_Gs_from_Hs(args, H_i)
	Gs_u,Gs_i = dict(),dict()
	Hs_u,Hs_i = dict(),dict()
	for key,val in G_u.items():
		Gs_u[key] = Tensor(G_u[key]).to(device)
		Hs_u[key] = Tensor(H_u[key]).to(device)
	for key,val in G_i.items():
		Gs_i[key] = Tensor(G_i[key]).to(device)
		Hs_i[key] = Tensor(H_i[key]).to(device)

	model = DualHGCN(in_ch=in_ft,n_hid=args.dim,dty_nets=Hs.keys(),inter=args.inter,intra=args.intra,dropout=args.dropout)
	model = model.to(device)
	emb = train(args, model, Xs_u, Xs_i, samples, Gs_u, Gs_i, Hs_u, Hs_i)
	return emb.detach().cpu().numpy()
	# return emb.detach().numpy()

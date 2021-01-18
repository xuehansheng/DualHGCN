#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from gensim.models import Word2Vec
from concurrent.futures import as_completed, ProcessPoolExecutor
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

### <!-- TiedAutoEncoder for generating initial features --!> ###
class TiedAutoEncoder(nn.Module):
	def __init__(self, inp, out):
		super().__init__()
		self.weight = nn.parameter.Parameter(torch.Tensor(out, inp))
		self.bias1 = nn.parameter.Parameter(torch.Tensor(out))
		self.bias2 = nn.parameter.Parameter(torch.Tensor(inp))
		
		self.register_parameter('tied weight',self.weight)
		self.register_parameter('tied bias1', self.bias1)
		self.register_parameter('tied bias2', self.bias2)
		
		self.reset_parameters()
		
	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias1 is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias1, -bound, bound)
		
		if self.bias2 is not None:
			fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_out)
			torch.nn.init.uniform_(self.bias2, -bound, bound)

	def forward(self, input):
		encoded_feats = F.linear(input, self.weight, self.bias1)
		encoded_feats = torch.tanh(encoded_feats)
		reconstructed_output = F.linear(encoded_feats, self.weight.t(), self.bias2)
		return encoded_feats, reconstructed_output

	def loss(self, y_pred, y_true):
		return torch.mean(torch.sum((y_true.ne(0).type(torch.float)*(y_true-y_pred))**2,dim=-1))

def train_tiedAE(A, dim, lr, weight_decay, n_epochs):
	tiedAE = TiedAutoEncoder(A.shape[-1], dim).to(device)
	optimizer = torch.optim.Adam(tiedAE.parameters(), lr=lr, weight_decay=weight_decay)
	A = torch.FloatTensor(A).to(device)

	for epoch in tqdm(range(n_epochs)):
		tiedAE.train()
		optimizer.zero_grad()
		encoded, reconed = tiedAE.forward(A)
		loss = tiedAE.loss(reconed,A)
		# print(loss)
		loss.backward()
		optimizer.step()
	return encoded.detach().numpy()
	# return encoded.cpu().detach().numpy()

import numpy as np
import sys
from scipy.optimize import minimize

def compute_flops(embed, mlp_ratio, seq_length, head, neuron_sparsity, head_sparsity, token_sparsity, TSL=[0,0,0,1,0,0,1,0,0,1,0,0]): # assume 12 blocks usually (deit - Ti, S, B)
	temp = 1 - token_sparsity
	token_sparsity = [0.]*3 + [1-temp]*3 + [1-temp**2]*3 + [1-temp**3]*3
	res = 0
	old_t = 0
	for n, h, t, s in zip(neuron_sparsity, head_sparsity, token_sparsity, TSL):
		if s == 0:
			# FFN
			res += 2 * int(seq_length*(1-t)) * embed * int(embed * mlp_ratio * (1-n))
			# MHSA
			chunk = embed / head
			per_head = 4 * embed * chunk * int(seq_length*(1-t)) + int(seq_length*(1-t))**2 * chunk * 2
			res += per_head * int(head * (1-h))
		else:
			# FFN
			res += 2 * int(seq_length*(1-t)) * embed * int(embed * mlp_ratio * (1-n))
			# MHSA
			chunk = embed / head
			per_head = 4 * embed * chunk * int(seq_length*(1-old_t)) + int(seq_length*(1-old_t))**2 * chunk * 2
			res += per_head * int(head * (1-h))
		old_t = t
	return res / 1e9

def compute_parameters(embed, mlp_ratio, seq_length, head, neuron_sparsity, head_sparsity):
	res = 0
	for n,h in zip(neuron_sparsity, head_sparsity):
		# FFN
		res += 2 * embed * int(embed * mlp_ratio * (1-n))
		# MHSA
		chunk = embed / head
		per_head = 4 * embed * chunk
		res += per_head * int(head * (1-h))
	res += embed * 1000 + 1000
	return res / 1e6






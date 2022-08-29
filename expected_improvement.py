import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from datasets import build_dataset
from losses import DistillationLoss
from samplers import RASampler
import models
import utils
import sys
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from engine import evaluate
from timm.utils import accuracy
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from compute_flops import compute_flops
from dependency_criterion import *
from scipy.optimize import NonlinearConstraint


num_blocks = 12


# nn.Linear indices
attn_qkv = [4*i+1 for i in range(num_blocks)]
attn_proj = [4*i+2 for i in range(num_blocks)]
mlp_fc1 = [4*i+3 for i in range(num_blocks)]
mlp_fc2 = [4*i+4 for i in range(num_blocks)]


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		if self.count > 0:
			self.avg = self.sum / self.count

	def accumulate(self, val, n=1):
		self.sum += val
		self.count += n
		if self.count > 0:
			self.avg = self.sum / self.count


@torch.no_grad()
def gp_evaluate(data_loader, model, device):
	criterion = torch.nn.CrossEntropyLoss()

	top1 = AverageMeter()

    # switch to evaluation mode
	model.eval()

	for _, (images, target) in enumerate(data_loader):
		images = images.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)

        # compute output
		with torch.cuda.amp.autocast():
			output = model(images, F.one_hot(target,1000))
			loss = criterion(output, target)

		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		batch_size = images.shape[0]
		top1.update(acc1.item(), batch_size)

	return top1.avg


def screen(flops_target, population, lb, ub, n_params, flops_mode):
	start_time = time.time()
	res = []
	while len(res) < population:
		ratio = (np.random.uniform(lb, ub, size=(1, n_params))[0]).tolist()
		if flops_mode == 'small':
			flops = compute_flops(384, 4, 197, 6, ratio[:12], ratio[12:24], ratio[-1]) # DEIT-S
		elif flops_mode == 'base':
			flops = compute_flops(768, 4, 197, 12, ratio[:12], ratio[12:24], ratio[-1]) # DEIT-B
		if abs(flops - flops_target) <= 0.02 * flops_target:
			if ratio not in res:
				res.append(ratio)
	print('Sampling time', time.time() - start_time)
	return res


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def random_point(flops_target, population, lb, ub, n_params, flops_mode):
	res = []
	while len(res) < population:
		ratio = (np.random.uniform(lb, ub, size=(1, n_params))[0]).tolist()
		if flops_mode == 'small':
			flops = compute_flops(384, 4, 197, 6, ratio[:12], ratio[12:24], ratio[-1]) # DEIT-S
		elif flops_mode == 'base':
			flops = compute_flops(768, 4, 197, 12, ratio[:12], ratio[12:24], ratio[-1]) # DEIT-B
		if abs(flops - flops_target) <= 0.02 * flops_target:
			if ratio not in res:
				res.append(ratio)
	return res[0]


def flops_constraint(x, embed=384, mlp_ratio=4, seq_length=197, head=6, TSL=[0,0,0,1,0,0,1,0,0,1,0,0]):
	neuron_sparsity = x[:12]
	head_sparsity = x[12:24]
	token_sparsity = x[-1]
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


def bayesian_optimisation(
    model, data_loader_val, n_params, n_iters, flops_target, neuron_rank, head_rank, device, population, lb, ub, flops_mode,
    gp_params=None, random_search=False,
    ):

	x_list = []
	y_list = []

	# initialize the population
	candidate_set = screen(flops_target=flops_target, population=population, lb=lb, ub=ub, n_params=n_params, flops_mode=flops_mode)
	for ratio in candidate_set:
		mlp_neuron_prune(model.module, mlp_neuron_mask(model.module, ratio[:12], neuron_rank))
		attn_head_prune(model.module, attn_head_mask(model.module, ratio[12:24], head_rank))
		set_token_selection_layer(model.module, ratio[-1])
		reward = gp_evaluate(data_loader_val, model, device)
		print('Neuron sparsity', ratio[:12])
		print('Head sparsity', ratio[12:24])
		print('Token sparsity', ratio[-1])
		print('Accuracy', reward)
		print('-'*100)
		mlp_neuron_restore(model.module)
		attn_head_restore(model.module)
		reset_token_selection_layer(model.module)

		x_list.append(np.array(ratio))
		y_list.append(reward)

	xp = np.array(x_list)
	yp = np.array(y_list)

	# Create the GP
	if gp_params is not None:
		gp_model = gp.GaussianProcessRegressor(**gp_params)
	else:
		kernel = gp.kernels.Matern()
		gp_model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10, normalize_y=False)

	for n in range(n_iters):

		# fit the GP model
		gp_model.fit(xp, yp)

		# Sample next hyperparameter
		if random_search:
			# x_random = np.random.uniform(bounds[:,0], bounds[:,1], size=(random_search, n_params))
			# expand search space by larger population size
			candidate_set = screen(flops_target=flops_target, population=population*100, lb=lb, ub=ub, n_params=n_params, flops_mode=flops_mode)
			x_random = []
			for ratio in candidate_set:
				x_random.append(np.array(ratio))
			x_random = np.array(x_random)
			ei = -1 * expected_improvement(x_random, gp_model, yp, greater_is_better=True, n_params=n_params)
			next_sample = x_random[np.argmax(ei), :]
		else:
			res = minimize(lambda x: -1 * expected_improvement(x, gp_model, yp, greater_is_better=True, n_params=n_params), 
				x0=random_point(flops_target=flops_target, population=1, lb=lb, ub=ub, n_params=n_params, flops_mode=flops_mode), 
				method='trust-constr',
				constraints=NonlinearConstraint(lambda x: flops_constraint(x), 0, flops_target))
			next_sample = res.x

		# Sample loss for new set of parameters
		mlp_neuron_prune(model.module, mlp_neuron_mask(model.module, next_sample.tolist()[:12], neuron_rank))
		attn_head_prune(model.module, attn_head_mask(model.module, next_sample.tolist()[12:24], head_rank))
		set_token_selection_layer(model.module, next_sample.tolist()[-1])
		reward = gp_evaluate(data_loader_val, model, device)
		print('Neuron sparsity', next_sample.tolist()[:12])
		print('Head sparsity', next_sample.tolist()[12:24])
		print('Token sparsity', next_sample.tolist()[-1])
		print('Accuracy', reward)
		print('-'*100)
		mlp_neuron_restore(model.module)
		attn_head_restore(model.module)
		reset_token_selection_layer(model.module)

		# Update lists
		x_list.append(next_sample)
		y_list.append(reward)

		# Update xp and yp
		xp = np.array(x_list)
		yp = np.array(y_list)

	return xp, yp
import torch
import numpy as np
from typing import Dict, Union

'''
Functions for calculating gradient of output with respect to input
'''


def generate_simple_gradients(
		model: torch.nn.Module,
		batch: Dict,
		inputs_embeds: torch.Tensor = None,
		differentiand='logits',
		aggregation='dot_product',
		apply_softmax=False):
	'''
	Calculate the gradient of the max class probability with respect to the inputs. Generate one gradient per input token by summing across the embedding vector for that token
	:param model:
	:param batch:
	:param with_respect_to: 'probs','logits', 'cross_entropy_loss'; what to differentiate. All is with respect to predicted class (e.g. probability of predicted class)
	:param aggregation: 'sum' or 'dot_product'; how to generate a single importance score for an embedding
	:param apply_softmax:
	:return:
	:param inputs_embeds: input embeddings; batch size x sequence length x embedding size
	:param py_probs: output class probabilities; batch size x num classes
	:return:
	'''


	if inputs_embeds is None:
		inputs_embeds = model.embed(batch['input_ids'])

	# batch['inputs_embeds'] = inputs_embeds

	with torch.enable_grad():
		model.train(True)
		inputs_embeds.requires_grad_(True)
		result = model(inputs_embeds=inputs_embeds, **batch)

		py_logits = result['py_logits']
		py_probs = result['py_probs']
		# model.zero_grad()
		if differentiand == 'logits':
			max_logits = torch.max(py_logits, dim=1)[0]
			grads = torch.autograd.grad(max_logits, inputs_embeds, only_inputs=True, grad_outputs=torch.ones_like(max_logits))[0]
		elif differentiand == 'probs':
			max_probs = torch.max(py_probs, dim=1)[0]
			grad_result = torch.autograd.grad(max_probs, inputs_embeds, only_inputs=True, grad_outputs=torch.ones_like(max_probs))
			grads = grad_result[0]
		elif differentiand == 'cross_entropy_loss':
			loss_vs_predicted = torch.nn.functional.cross_entropy(result['py_logits'], result['py_index'], reduction='none')
			grad_result = torch.autograd.grad(loss_vs_predicted, inputs_embeds, only_inputs=True, grad_outputs=torch.ones_like(loss_vs_predicted))
			grads = grad_result[0]

		if aggregation == 'sum':
			aggregated_grads = torch.sum(grads.abs(), dim=2)
		elif aggregation == 'dot_product':
			aggregated_grads = torch.sum(grads*inputs_embeds,dim=2)

		if apply_softmax:
			aggregated_grads = torch.nn.functional.softmax(aggregated_grads, dim=1)

		aggregated_grads = aggregated_grads.detach()
		model.zero_grad()
		model.train(False)

	return aggregated_grads


def generate_integrated_gradients(model: torch.nn.Module,
								  batch: Dict,
								  baseline_token_id: Union[int,str],
								  num_intervals=10):
	'''
	Calculate the integrated gradient (TKTK) of the max class probability with respect to the input.

	Basically just calculates simple gradients at even intervals between a baseline input and the given input,
	then calculates the AUC under that curve

	Supposedly has some nice axiomatic properties.
	:param model:
	:param inputs_embeds:
	:param py_probs:
	:return:
	'''

	inputs_embeds = model.embed(batch['input_ids'])

	if type(baseline_token_id) == int:
		baseline_input_ids = torch.ones_like(batch['input_ids']) * baseline_token_id
		baseline_inputs_embeds = model.embed(baseline_input_ids)
	elif baseline_token_id == 'zero':
		baseline_inputs_embeds = torch.zeros_like(inputs_embeds)

	intervals = torch.arange(0, 1 + 1 / num_intervals, 1 / num_intervals, device=inputs_embeds.device)

	interval_grad_list = []
	for baseline_weight in intervals:
		interval_embeds = baseline_weight * baseline_inputs_embeds + (1 - baseline_weight) * inputs_embeds
		interval_grads = generate_simple_gradients(model=model,
												   batch=batch,
												   inputs_embeds=interval_embeds)
		interval_grad_list.append(interval_grads)

	# even intervals from 0 to 1, so the area under the curve is just the mean value
	auc_grads = torch.stack(interval_grad_list, dim=2).mean(dim=2)
	grads = torch.stack(interval_grad_list, dim=2)
	auc_grads = torch.sum((intervals[1:] - intervals[:-1]) * ((grads[:,:,:-1] + grads[:,:,1:])/2), dim=2)

	return auc_grads

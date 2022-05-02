import math
from typing import Dict, List

import numpy as np
import torch


def add_random_rationales(result:Dict, probabilities:List):

	if probabilities is not None:
		for probability in probabilities:
			probs = torch.ones_like(result['input_ids']) * probability
			rationale = torch.bernoulli(probs).detach()
			result[f'rationale_random_{int(100*probability)}'] = rationale


def add_rationale_permutations(result, percentages):
	'''
	result is a dictionary containing the output of forward()
	notable keys include:
	'rationale': human rationale
	'padding_mask': mask indicating text
	'special_mask': mask indicating special tokens that probably shouldn't be occluded
	:param result:
	:param percentages:
	:return:
	'''
	print("Perturbing")
	dropped_masks = [[] for p in percentages]
	permuted_masks = [[] for p in percentages]
	for text,row in zip(result["padding_mask"],result["human_rationale"]):
		rationale_indices = row.nonzero(as_tuple=True)[0].cpu()
		rationale_length = rationale_indices.size(0)
		non_rationale_indices = (text - row).nonzero(as_tuple=True)[0].cpu()
		non_rationale_length = non_rationale_indices.size(0)
		indices = torch.randperm(rationale_length)
		non_rationale_perm_indices = torch.randperm(non_rationale_length)
		for i,p in enumerate(percentages):
			new_row = np.ones(row.size(0))
			flip_indices = rationale_indices[indices[:math.ceil(p * rationale_length)]]
			new_row[flip_indices] = 0
			# dropped_masks[i].append(torch.from_numpy(new_row).to(row.device) * row)
			non_rationale_flip_indices = non_rationale_indices[non_rationale_perm_indices[:math.ceil(p * rationale_length)]]
			non_rationale_new_row = np.zeros(row.size(0))
			non_rationale_new_row[non_rationale_flip_indices] = 1
			permuted_masks[i].append(torch.from_numpy(new_row).to(row.device) * row + torch.from_numpy(non_rationale_new_row).to(row.device))
	for i,p in enumerate(percentages):
		# result[f'rationale_dropped_{str(p * 100).replace(".0","")}%'] = torch.stack(dropped_masks[i],dim=0)
		result[f'human_rationale_permuted_{str(p * 100).replace(".0","")}%'] = torch.stack(permuted_masks[i],dim=0)
	return result


def test_add_rationale_permutations():
	pass
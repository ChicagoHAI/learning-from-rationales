from typing import List

import torch


def batch_pad_2d(tensors: List[torch.Tensor], pad_value: torch.Tensor, dim: int, return_padding_mask=True, max_length: int = None):
	'''
	Pad a series of 2d
	:param sequences:
	:param pad_value:
	:param return_padding_mask:
	:return:
	'''
	if max_length is None:
		max_length = max(s.shape[dim] for s in tensors)

	padded_tensors = []
	for i, s in enumerate(tensors):
		if s.ndim < 2: s = s.unsqueeze(0)
		padded_sequence = torch.cat([s] + (max_length - s.shape[0]) * [pad_value], dim=dim)
		padded_tensors.append(padded_sequence)

	padded = torch.stack(padded_tensors, dim=dim)

	if not return_padding_mask:
		return padded
	else:
		padding_mask = torch.stack([torch.nn.functional.pad(torch.ones(s.shape[0], dtype=torch.int, device=s.device), (0, max_length - len(s)), value=0) for s in tensors], dim=0)
		return padded, padding_mask
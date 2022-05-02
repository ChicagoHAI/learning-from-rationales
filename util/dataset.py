from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from util.print_util import iprint

from itertools import repeat
from typing import Sequence, Union, Any, List
import pandas.core.common as com


class RationaleDataset(Dataset):
	def __init__(self,
				 input_ids=None,
				 human_rationale=None,
				 human_rationale_weight=None,
				 special_mask=None,
				 label=None,
				 p_alphas: Dict = None,
				 sentence_ids=None,
				 pseudoexample=None,
				 token_type_ids=None,
				 batch_width: Union[str, int] = None):

		self.input_ids = [torch.tensor(i) for i in input_ids]
		self.max_length = max(seq.shape[0] for seq in self.input_ids)

		self.human_rationale = [torch.tensor(r) for r in human_rationale]
		self.human_rationale_weight = [torch.tensor(w) for w in human_rationale_weight]
		self.special_mask = [torch.tensor(m) for m in special_mask]
		self.has_label = not np.all(pd.isnull(label))

		if self.has_label:
			self.label = torch.tensor(label)
		else:
			self.label = None

		if p_alphas is not None:
			self.p_alphas = {key: [torch.tensor[p] for p in p_alphas[key]] for key in p_alphas}
		else:
			self.p_alphas = {}

		if sentence_ids is not None:
			self.sentence_ids = [torch.tensor(s) for s in sentence_ids]
		else:
			self.sentence_ids = None

		if token_type_ids is not None:
			self.token_type_ids = [torch.tensor(s) for s in token_type_ids]
		else:
			self.token_type_ids = None

		self.pseudoexample = pseudoexample

		# set the batch width to a constant, the global max length, or the local max length for each batch
		if type(batch_width) == int or batch_width == 'local_max':
			self.batch_width = batch_width
		elif batch_width == 'global_max':
			self.batch_width = self.max_length

	# def tile(self, num):
	# 	iprint(f'Tiling dataset by factor of {num} for benchmarking purposes. {len(self)} items present initially.')
	# 	self.input_ids = [t for i in range(num) for t in self.input_ids]
	# 	self.human_rationale = [t for i in range(num) for t in self.human_rationale]
	# 	self.human_rationale_weight=[t for i in range(num) for t in self.human_rationale_weight]
	# 	self.special_mask = [t for i in range(num) for t in self.special_mask]
	# 	if self.has_label: self.label = [t for i in range(num) for t in self.label]
	#
	# 	self.p_alphas = {key: [t for i in range(num) for t in p_alphas] for key, p_alphas in self.p_alphas.items()}
	#
	# 	iprint(f'{len(self)} items in dataset after tiling.')
	# 	return

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		rdict = {
			'input_ids': self.input_ids[idx],
			'human_rationale': self.human_rationale[idx],
			'human_rationale_weight': self.human_rationale_weight[idx],
			'special_mask': self.special_mask[idx],
		}
		if self.has_label:
			rdict['label'] = self.label[idx]

		if self.p_alphas is not None:
			rdict.update({key: self.p_alphas[key][idx] for key in self.p_alphas})

		if self.sentence_ids is not None:
			rdict['sentence_ids'] = self.sentence_ids[idx]

		if self.pseudoexample is not None:
			rdict['pseudoexample'] = self.pseudoexample[idx]

		if self.token_type_ids is not None:
			rdict['token_type_ids'] = self.token_type_ids[idx]

		return rdict

	def sample(self, n: int = None, random_state: int = None, replace: bool = False, weights: Sequence = None):
		'''
		Subsample this dataset. Should have identical behavior to the pandas DataFrame.sample method
		:param n:
		:param random_state:
		:param replace:
		:param weights:
		:return:
		'''
		rs = com.random_state(random_state)
		indices = rs.choice(self.__len__(), size=n, replace=replace, p=weights)
		sampled = RationaleDataset(
			input_ids=nonesafe_sample(self.input_ids, indices),
			human_rationale= nonesafe_sample(self.human_rationale, indices),
			human_rationale_weight=nonesafe_sample(self.human_rationale_weight, indices),
			special_mask=nonesafe_sample(self.special_mask, indices),
			label=nonesafe_sample(self.label, indices),
			p_alphas={key:nonesafe_sample(val, indices) for key, val in self.p_alphas.items()},
			sentence_ids=nonesafe_sample(self.sentence_ids, indices),
			pseudoexample=nonesafe_sample(self.pseudoexample, indices),
			token_type_ids=nonesafe_sample(self.token_type_ids, indices),
			batch_width=self.batch_width

		)
		return sampled


def nonesafe_sample(s: List[Any], indices: Sequence[int]):
	if s is None:
		return None
	else:
		return [s[i] for i in indices]

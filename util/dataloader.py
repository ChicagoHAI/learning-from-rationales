import numpy as np
import torch
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import Sampler, DataLoader, Dataset

from util.data_util import batch_pad_1d, tensor_dict_to_numpy, invert_dict_list, combine_results_dict
from util.dataset import RationaleDataset
from util.print_util import iprint


class MaxSampler(Sampler):
	'''
	A sampler that only ever samples the biggest batch. Useful for tuning the batch size.
	'''

	def __init__(self, *args, data_source=None, **kwargs):
		super().__init__(*args, data_source=data_source,**kwargs)
		self.sizes = [sum([value.numel() for value in item.values()]) for item in data_source]
		self.max_idx = np.argmax(self.sizes)
		iprint(f'Max width: {data_source[self.max_idx]["input_ids"].shape[0]}')
		pass

	def __iter__(self):
		for size in self.sizes:
			yield self.max_idx

	def __len__(self):
		return len(self.sizes)




class RationaleDataloader(DataLoader):
	'''
	Support for variable-width batch dataloaders doesn't come standard in pytorch for some reason
	'''
	def __init__(self,
				 dataset: RationaleDataset,
				 pad_token_id:int,
				 description:str='',
				 **kwargs):
		super().__init__(dataset, **kwargs)
		# self.dataset = dataset
		self.pad_token_id=pad_token_id
		self.collate_fn = self.collate
		self.max_length = dataset.max_length
		self.has_label = dataset.has_label
		self.description=description
		self.batch_width = dataset.batch_width

	def collate(self, items, *args, **kwargs):
		padded_input_ids, padding_mask = batch_pad_1d([i['input_ids'] for i in items], self.pad_token_id, return_padding_mask=True, max_length=self.batch_width)
		batch= {'input_ids':padded_input_ids,
				'padding_mask':padding_mask,
				'human_rationale':batch_pad_1d([i['human_rationale'] for i in items], 0.0, max_length=self.batch_width),
				'human_rationale_weight':batch_pad_1d([i['human_rationale_weight'] for i in items], 0.0, max_length=self.batch_width),
				'special_mask':batch_pad_1d([i['special_mask'] for i in items], 0.0, max_length=self.batch_width),

				}

		if self.has_label:
			batch['label'] = torch.stack([i['label'] for i in items])

		if self.dataset.p_alphas is not None:
			batch.update({key:batch_pad_1d([i[key] for i in items], 0.0, max_length=self.batch_width) for key in self.dataset.p_alphas})

		if self.dataset.sentence_ids is not None:
			batch['sentence_ids'] = batch_pad_1d([i['sentence_ids'] for i in items], 0.0, max_length=self.batch_width)

		if self.dataset.pseudoexample is not None:
			batch['pseudoexample'] =torch.tensor([i['pseudoexample'] for i in items])

		if self.dataset.token_type_ids is not None:
			batch['token_type_ids'] = batch_pad_1d([i['token_type_ids'] for i in items], 0, max_length=self.batch_width)


		return batch





def create_dataloader(dataset: Dataset, max_only=False, num_workers=0, batch_size=None, pad_token_id=None):

	dataloader= RationaleDataloader(
		dataset,
		pad_token_id=pad_token_id,
		batch_size=batch_size,
		shuffle=not max_only,
		# num_workers=num_workers,
		pin_memory=True,
		sampler=MaxSampler(data_source=dataset) if max_only else None,
		# multiprocessing_context = 'fork' if num_workers > 0 else None
	)
	return dataloader
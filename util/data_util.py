'''
Data manipulation utils. Primarily custom pytorch Dataset and Dataloader for dealing with the rationale data
'''

import torch
from typing import List, Dict
import numpy as np

# from util.dataloader import MaxSampler, RationaleDataloader
from util.print_util import iprint
import pandas as pd


def batch_pad_1d(sequences:List[torch.Tensor], pad_value, return_padding_mask=False, max_length='auto'):
	'''
	Pad a list of variable length 1D tensors to the max length and then stack them into a single 2D tensor
	:param sequences:
	:param pad_value:
	:return:
	'''
	if max_length == 'auto' or max_length == 'local_auto':
		max_length = max(s.shape[0] for s in sequences)

	padded = torch.stack([torch.nn.functional.pad(s,(0,max_length-len(s)), value=pad_value) for s in sequences], dim=0)

	if not return_padding_mask:
		return padded
	else:
		padding_mask = torch.stack([torch.nn.functional.pad(torch.ones(s.shape[0],dtype=torch.int,device=s.device),(0,max_length-len(s)), value=0) for s in sequences], dim=0)
		return padded, padding_mask


def batch_unpad(batch: np.ndarray, padding_mask: np.ndarray):
	'''
	Unpad a numpy array into a list of variable-length 1d numpy arrays using a padding mask
	:param batch:
	:param padding_mask:
	:return:
	'''

	unpadded_rows = []
	for batch_row, padding_row in zip(batch, padding_mask):
		unpadded_rows.append(batch_row[np.where(padding_row == 1)[0]])

	return unpadded_rows


def combine_results_dict(results_dict:Dict[str,List], verbose=False):
	'''
	Take a dictionary of (key --> list of batch outputs), as produced by invert_dict_list, and concatenate the inner lists into sensible data objects
	:param results_dict:
	:return:
	'''

	combined_results_dict = {}
	for key in results_dict.keys():
		# if verbose: iprint(key)
		if results_dict[key][0].ndim > 2: #Just ignore 3D+ items
			if verbose: iprint(f'Skipping {key} because it is {results_dict[key][0].ndim}D')
			continue
		elif results_dict[key][0].ndim == 0: #1D or 0D are easy--just concatenate
			combined_results_dict[key] = np.array(results_dict[key])
		elif results_dict[key][0].ndim == 1:
			combined_results_dict[key] = np.concatenate(results_dict[key])
		elif results_dict[key][0].ndim == 2: #2D items may be of varying batch widths because of the variable size padding above.
			all_same_width = all([t.shape[1] == results_dict[key][0].shape[1] for t in results_dict[key]])
			resolved = False
			#If there is a padding mask available and it matches the size of the outputs, then use it to unpad
			if 'padding_mask' in results_dict and key != 'padding_mask':
				values_and_padding = list(zip(results_dict[key], results_dict['padding_mask']))
				if all(t.shape == p.shape for t, p in values_and_padding):
					combined_results_dict[key] = []
					for t, p in values_and_padding:
						combined_results_dict[key].extend(batch_unpad(t, p))
					resolved=True
				elif not all_same_width:
					if verbose:iprint(f'Skipping {key} because batch sizes were variable and did not match padding masks')
					continue

			#Otherwise, if all the batches are the same width, just stack them
			if not resolved and all_same_width:
				combined_results_dict[key] = np.vstack(results_dict[key])
			elif not resolved: # Otherwise we don't have a good way of combining the outputs, so skip
				# We could try to automatically figure out how to unpad them, but the better solution is to just provide the padding mask in the first place
				if verbose: iprint(f'Skipping {key} because batch sizes were variable and no padding masks were available or key is "padding_mask"')
				continue

	# if verbose:
	# 	iprint(f'Shape: {np.shape(combined_results_dict[key])}')


	return combined_results_dict

def invert_dict_list(dict_list:List[Dict]):
	'''
	Convert a list of dictionaries into a dictionary of lists
	:param dict_list:
	:return:
	'''
	list_dict = {}
	for dict_element in dict_list:
		for key in dict_element.keys():
			if key not in list_dict:
				list_dict[key] = []
			list_dict[key].append(dict_element[key])

	return list_dict

def detach_tensor_dict(result_dict:Dict):
	return {key:value.detach() for key, value in result_dict.items()}

def tensor_dict_to_numpy(result_dict:Dict):
	return_dict = {}
	for key in result_dict.keys():
		if result_dict[key] is not None:
			return_dict[key] = result_dict[key].detach().cpu().numpy()
	return return_dict

def tensor_dict_to_cuda(result_dict:Dict):
	return_dict = {}
	for key in result_dict.keys():
		return_dict[key] = result_dict[key].cuda()
	return return_dict


def model_output_to_df(output_dict:Dict, method:str):
	'''
	Convert a combined output dictionary (as produced by run_model_on_dataloader()) to a dataframe.
	:param output_dict:
	:param method:
	:return:
	'''
	if method == 'max':
		# Only get outputs of the max length in the dictionary (meaning example-wise outputs like predicted label)
		max_output_length = max([len(value) for value in output_dict.values()])
		max_dict = {key:value for key, value in output_dict.items() if len(value) == max_output_length}
		for key in max_dict.keys():
			if np.ndim(max_dict[key]) > 1:
				max_dict[key] = list(max_dict[key])

		output_df = pd.DataFrame(max_dict)
		return output_df
	else:
		raise Exception(f'Unknown method {method}')


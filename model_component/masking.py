from typing import Callable

import torch
from transformers import BertModel

'''
Utility functions for masking
'''


class MaskInput(torch.nn.Module):

	def forward(self, **kwargs):
		return mask_input(**kwargs)



def mask_input(
		inputs_embeds: torch.Tensor = None,
		input_ids: torch.Tensor = None,
		mask: torch.Tensor = None,
		masking_strategy: str = None,
		padding_mask: torch.Tensor = None,
		special_mask: torch.Tensor = None,
		mask_token_id: torch.Tensor = None,
		pad_token_id: torch.Tensor = None,
		word_embedding_function: Callable = None,
		rationale_embedding_function:Callable=None,
):
	'''
	Take a batch size x batch width input ID or batch size x batch width x embedding size embedding tensor,
	and mask it with a batch size x batch width mask tensor.

	Complicated by the fact that some masking (token removal) should be applied to input IDs while other masking
	should be applied to input embeddings (multiplicative masking).

	Returns a dictionary consisting of input embeddings, input token IDS, padding mask, special token mask,
	and optionally token type IDS, all adjusted to account for masking when appropriate.

	Using the term "extras" to refer to whatever is given of padding_mask, special_mask and token_type_IDS

	Pseudocode:
	If the masking strategy is "removal"
		Perform the removal on input_ids and extras
		Apply embedding function to masked input_ids
		Return masked input ids, embeddings and extras
	If the masking strategy is "multiply_mask" or "multiply_zero"
		If embeddings not provided, apply embedding function to input_ids
		Perform masking multiply on embeddings
		Extras remain unchanged
		Binarize mask and use it to multiplicatively mask input_ids, just for bookkeeping/sanity checking
		Return masked input_ids, embeddings and unchanged extras
	If the masking strategy is "bert_attention"
		Multiply padding_mask by mask to get masked padding mask
		Input_ids and extras remain unchanged
		If embeddings not provided, generate embeddings from input ids
		Return input_ids, embeddings, masked padding mask and other extras
	If the masking strategy is "token_type_ids"
		If embeddings not provided, generate embeddings from input ids
		Subtract the 0 token type embedding from the whole embedding matrix to undo that operation from the embedding function
		Perform additive masking using the built-in token type ID embedding matrix
		return input_ids, masked embeddings, extras
	If the masking strategy is "0_1_embeddings" or "embeddings":
		If embeddings not provided, generate embeddings from input ids
		Perform additive masking using the manually-created embedding matrix (which should be appropriate to the masking strategy)
		return input_ids, masked embeddings, extras


	:param inputs_embeds:
	:param mask:
	:param masking_strategy:
	:param mask_embedding:
	:param pad_embedding:
	:return:
	'''


	safe_embed_function = lambda t:word_embedding_function(t) if word_embedding_function is not None else None

	if mask is None:
		if inputs_embeds is None and input_ids is not None:
			inputs_embeds = safe_embed_function(input_ids)
		result =  {
			'masked_inputs_embeds':inputs_embeds,
			'masked_input_ids':input_ids,
			'masked_padding_mask':padding_mask,
			'masked_special_mask':special_mask
		}

	else:
		if masking_strategy == 'removal':
			# remove masked token ids and shift other ones left, then produce embeddings
			masked_input_ids, masked_padding_mask, masked_special_mask = mask_input_by_removal(input_ids=input_ids,
																							   mask=mask.detach(),
																							   pad_token_id=pad_token_id,
																							   padding_mask=padding_mask,
																							   special_mask=special_mask)
			masked_inputs_embeds = safe_embed_function(masked_input_ids)
			result =  {
				'masked_inputs_embeds':masked_inputs_embeds,
				'masked_input_ids':masked_input_ids,
				'masked_padding_mask':masked_padding_mask,
				'masked_special_mask':masked_special_mask
			}


		else:
			if inputs_embeds is None:
				inputs_embeds = safe_embed_function(input_ids)

			result = mask_embeddings(
				inputs_embeds=inputs_embeds,
				mask=mask,
				padding_mask=padding_mask,
				word_embedding_function=word_embedding_function,
				mask_token_id=mask_token_id,
				rationale_embedding_function=rationale_embedding_function,
				masking_strategy=masking_strategy
			)
			result.update(
				{
					'masked_input_ids':input_ids,
					'masked_padding_mask':special_mask,
				}
			)


	return result

def mask_embeddings(inputs_embeds:torch.Tensor,
					mask:torch.Tensor,
					padding_mask:torch.Tensor,
					word_embedding_function:Callable,
					mask_token_id: torch.Tensor,
					rationale_embedding_function:Callable,
					masking_strategy:str,
					):
	'''
	Do masking on a tensor of token embeddings

	:param inputs_embeds:
	:param mask:
	:param padding_mask:
	:param word_embedding_function:
	:param mask_token_id:
	:param rationale_embedding_function:
	:param masking_strategy:
	:return:
	'''


	if masking_strategy == 'multiply_zero':
		# Just zero out the masked token embeddings
		masked_inputs_embeds = inputs_embeds * mask.unsqueeze(2)
		# masked_input_ids = input_ids
		masked_padding_mask = padding_mask
	# masked_special_mask = special_mask
	elif masking_strategy == 'multiply_mask':
		# Replace masked token embeddings with the embedding for the [MASK] token
		# mask_token_ids = torch.ones_like(mask) * mask_token_id
		# with torch.no_grad():
		# 	mask_embeds = word_embedding_function(mask_token_ids)
		mask_embedding = word_embedding_function(torch.scalar_tensor(mask_token_id,dtype=torch.long,device=mask.device))
		masked_inputs_embeds = inputs_embeds * mask.unsqueeze(2) + mask_embedding * (1 - mask.unsqueeze(2))
		# rounded_mask = (mask >= 0.5).int()
		# masked_input_ids = input_ids * rounded_mask + mask_token_ids * (1 - rounded_mask)
		masked_padding_mask = padding_mask
	# masked_special_mask = special_mask

	elif masking_strategy == 'bert_attention':
		#Just apply mask to the padding mask, so that bert ignores the indicated tokens
		masked_inputs_embeds = inputs_embeds
		# masked_input_ids = input_ids
		# masked_padding_mask = padding_mask * mask
		masked_padding_mask = padding_mask - (1-mask)

	# masked_special_mask = special_mask
	elif masking_strategy == 'token_type_ids':
		zero_embeddings = rationale_embedding_function(torch.zeros_like(mask,dtype=torch.long))
		type_embeddings = zero_embeddings * (1 - mask).unsqueeze(2) + \
						  rationale_embedding_function(torch.ones_like(mask,dtype=torch.long)) * mask.unsqueeze(2)
		#the subtraction is needed because the BERT embedding layer adds it by default
		masked_inputs_embeds = inputs_embeds - zero_embeddings+ type_embeddings
		# masked_inputs_embeds = inputs_embeds - type_embeddings

		# masked_input_ids = input_ids
		masked_padding_mask = padding_mask
	# masked_special_mask = special_mask
	elif masking_strategy == '0_1_embeddings' or masking_strategy == 'embeddings':
		type_embeddings = rationale_embedding_function(torch.zeros_like(mask)) * (1 - mask).unsqueeze(2) + rationale_embedding_function(torch.ones_like(mask)) * mask.unsqueeze(2)
		masked_inputs_embeds = inputs_embeds + type_embeddings
		# masked_input_ids = input_ids
		masked_padding_mask = padding_mask
	# masked_special_mask = special_mask
	else:
		raise Exception(f'Unknown masking strategy {masking_strategy}')

	result ={'masked_inputs_embeds':masked_inputs_embeds,
			 'masked_padding_mask':masked_padding_mask}

	return result

def mask_input_by_removal(input_ids: torch.Tensor,
						  mask: torch.Tensor,
						  pad_token_id: int,
						  padding_mask:torch.Tensor=None,
						  special_mask:torch.Tensor=None):
	'''
	Mask an input tensor with a 0-1 mask by removing masked tokens and shifting remaining tokens to the left and substituting [PADDING] tokens as needed.
	Return an updated padding mask and/or updated special token mask

	For now, not allowing 3d input (i.e. embedding tensors). There's nothing impossible about doing removal on the embedding tensor, but it will
	be inefficient and there's no point, since it's a non-differentiable operation.

	param input_ids
	:param mask: 2d binary mask tensor
	:param padding_mask: 2d binary mask tensor indicating existing masking
	:param pad_token_id: scalar or 1d tensor
	:return:
	'''



	assert input_ids.ndim == 2

	# mm = mask.float().mean()
	# if mm > 0 and mm < 1:
	# 	x=0


	lengths = tuple(mask.sum(dim=1).int().detach().cpu().numpy())
	shortened_input_id_rows = input_ids[mask.bool()].split(lengths)
	padded_shortened_input_id_rows = batch_pad_1d(shortened_input_id_rows, pad_token_id, return_padding_mask=False, max_length=input_ids.shape[1])

	other_masks = [padding_mask, special_mask]
	# if len(other_masks) > 0:
	shortened_other_masks = [other_mask[mask.bool()].split(lengths) if other_mask is not None else None for other_mask in other_masks]
	padded_shortened_other_masks = [batch_pad_1d(shortened_other_mask, 0, return_padding_mask=False, max_length=input_ids.shape[1]) if shortened_other_mask is not None else None for shortened_other_mask in shortened_other_masks]
	return [padded_shortened_input_id_rows] + padded_shortened_other_masks
	# else:
	# 	return padded_shortened_input_id_rows


# row_indices = [torch.nonzero(mask_row).squeeze(1) for mask_row in mask]

	# shortened_input_id_rows = [input_row[indices] for input_row, indices in zip(input_ids, row_indices)]
	# shortened_other_masks = [[other_mask_row[indices] for other_mask_row, indices in zip(other_mask, row_indices)] if other_mask is not None else None for other_mask in other_masks ] if other_masks is not None else []


	# shortened_rows = [embedding_row[torch.nonzero(mask_row).squeeze(1)] for embedding_row, mask_row in zip(inputs, mask * padding_mask)]
	# shortened_rows = []
	# for embedding_row, mask_row in zip(inputs, mask * padding_mask):
	# 	shortened_row = embedding_row[torch.nonzero(mask_row).squeeze()]
	# 	if shortened_row.ndim() ==

	# if inputs.ndim == 3:
	# 	padded_shortened_rows, new_padding_mask = batch_pad_2d(shortened_rows, padding_token.squeeze(0), dim=0, return_padding_mask=True, max_length=inputs.shape[1])
	# else:
	# 	padded_shortened_rows, new_padding_mask = batch_pad_1d(shortened_rows, padding_token, return_padding_mask=True, max_length=inputs.shape[1])


from util.data_util import batch_pad_1d
import numpy as np


def test_mask_by_removal():
	documents = ['cows are ungulate mammals',
				 'mars is the fourth planet from the sun',
				 'phillipe petain was a world war 1 general']

	queries = ['cows give live birth',
			   'mars is an asteroid',
			   'phillipe petain fought in world war 1']

	sequences = [np.array(['[CLS]'] + document.split() + ['[SEP]'] + query.split() + ['[SEP]']) for document, query in zip(documents, queries)]
	special_masks = [torch.tensor([1] + [0] * len(document.split()) + [1] + [1] * len(query.split()) + [1]) for document, query in zip(documents, queries)]

	special_tokens = ['[PAD]', '[SEP]', '[CLS]']
	unique_tokens = list(set([token for sequence in sequences for token in sequence if token not in special_tokens]))
	vocab = special_tokens + sorted(unique_tokens)
	vocab_dict = {token: idx for idx, token in enumerate(vocab)}
	pad_token_id = vocab_dict['[PAD]']

	token_ids = [torch.tensor([vocab_dict[token] for token in sequence]) for sequence in sequences]

	input_ids, padding_mask = batch_pad_1d(token_ids, pad_token_id, return_padding_mask=True)
	special_mask = batch_pad_1d(special_masks, 0, return_padding_mask=False)

	assert torch.all(input_ids == torch.tensor([[2, 9, 6, 25, 18, 1, 9, 14, 17, 8, 1, 0, 0, 0, 0, 0, 0, 0],
												[2, 19, 16, 24, 11, 22, 12, 24, 23, 1, 19, 16, 5, 7, 1, 0, 0, 0],
												[2, 21, 20, 27, 4, 28, 26, 3, 13, 1, 21, 20, 10, 15, 28, 26, 3, 1]]))

	assert torch.all(padding_mask == torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
												   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
												   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
												  dtype=torch.int32))

	assert torch.all(special_mask == torch.tensor([[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
												   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
												   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

	p_alpha = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]])

	p_alpha = 1 - ((1 - p_alpha) * (1 - special_mask))  # apply special mask to p_alpha

	masked_input_ids, masked_padding_mask, masked_special_mask = mask_input_by_removal(input_ids=input_ids,
																					   mask=p_alpha,
																					   pad_token_id=pad_token_id,
																					   other_masks=[padding_mask, special_mask])
	np.array([[vocab[id.numpy().item()] for id in ids] for ids in masked_input_ids])

	return

	# np.random.seed(64646)
	# embeddings = np.array([[i // 10, i] for i in range(len(vocab))])
	#
	# pad_embedding = torch.tensor(embeddings[vocab_dict['[PAD]']])
	# token_embeddings = torch.tensor([[embeddings[token_id] for token_id in id_sequence] for id_sequence in padded_token_ids])
	#
	# mask = torch.tensor([[0, 0, 0, 1, 1, 1],  # sprinkle random ones in places where there is padding
	# 					 [1, 0, 0, 1, 1, 0],
	# 					 [1, 1, 0, 0, 1, 0]])
	#
	# masked_input, new_padding_mask = mask_input(token_embeddings, mask, 'removal', padding_mask, pad_embedding=pad_embedding)

	# collapsed_masked_input =



def calculate_sparsity_loss(p_alpha=None, padding_mask=None):
	sparsity_loss = masked_mean(p_alpha, padding_mask, dim=1).mean()
	cohesiveness_loss = masked_mean(torch.abs(p_alpha[:, 0:-1] - p_alpha[:, 1:]), padding_mask[:, 0:-1], dim=1).mean()
	return sparsity_loss, cohesiveness_loss


class MaskedMean(torch.nn.Module):
	'''
	Wrap masked_mean in a class to simplify TensorBoard graph diagrams
	'''
	def forward(self, *args, **kwargs):
		return masked_mean(*args, **kwargs)

def masked_mean(t: torch.Tensor = None, mask: torch.Tensor = None, dim: int = -1, keepdim: bool = False):
	'''
	Adapted from AllenNLP. Returns a mean across a tensor, excluding certain elements from the calculation
	:param t:
	:param mask:
	:param dim:
	:param keepdim:
	:return:
	'''
	replaced_vector = t.masked_fill(~mask.bool(), 0.0)
	value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
	value_count = torch.sum(mask, dim=dim, keepdim=keepdim)

	mean_value = value_sum / value_count.float().clamp(min=1e-13)

	return mean_value


def masked_sum(t: torch.Tensor = None, mask: torch.Tensor = None, dim: int = -1, keepdim: bool = False):
	'''
	:param t:
	:param mask:
	:param dim:
	:param keepdim:
	:return:
	'''
	replaced_vector = t.masked_fill(~mask.bool(), 0.0)
	value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)

	return value_sum


def process_py_logits(py_logits: torch.Tensor = None):
	'''
	Calculate predicted class and class probs from class logits
	:param py_logits:
	:return:
	'''
	py_index = torch.max(py_logits, 1).indices
	py_probs = torch.nn.functional.softmax(py_logits, dim=1)
	return py_index, py_probs


def masked_flatten(t: torch.Tensor, mask: torch.Tensor):
	'''
	Flatten a 2d tensor to 1d, dropping values as indicated by the mask
	:param t:
	:param mask:
	:return:
	'''

	# t1 = t.view(-1)
	# mask1 = mask.view(-1)

	# return t1[torch.nonzero(mask1)][:, 0]


	return t[mask.bool()]


def main():
	test_mask_by_removal()


if __name__ == '__main__':
	main()


def create_rationale_embedding_function(bert_model:BertModel=None, masking_strategy:str=None, embedding_size:int=None):
	'''
	Create a function which will take in a binary rationale matrix and return
	rationale embeddings. This is needed for the three embedding-based masking strategies
	:param bert_model:
	:param masking_strategy:
	:return:
	'''


	if masking_strategy == '0_1_embeddings': #use an "embedding" matrix where the embedding is just  [[... 0 0 1 0],[... 0 0 0 1]]
		embedding_weight = torch.zeros((2, embedding_size))
		embedding_weight[0,-2]=1.0
		embedding_weight[1,-1]=1.0
		return torch.nn.Embedding(num_embeddings=2, embedding_dim= embedding_size, _weight=embedding_weight)
	elif masking_strategy == 'embeddings': # use a new embedding matrix initialized the same way as the BertEmbeddings
		return torch.nn.Embedding(num_embeddings=2, embedding_dim= embedding_size)

	else:
		return lambda:None
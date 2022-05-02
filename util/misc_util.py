import multiprocessing as mp
import traceback

from util.print_util import iprint
import numpy as np
import re
import torch
import random
import os
from datetime import datetime
import pandas as pd
from typing import Dict, Union, List, Callable, Tuple
# import intervaltree as it

def set_random_seeds(seed, n_gpu=1):
	iprint(f'Setting python, numpy and torch seeds to {seed}')
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(seed)


def binarize(y, threshold=0.5):
	if type(y) == list:
		return [1.0 if yi >= threshold else 0.0 for yi in y]
	else:
		if hasattr(y, 'astype'):
			return (y >= threshold).astype(np.float32)
		else:
			return float(y >= threshold)


nonnumeric = re.compile('[^0-9\.]+')


def process_number_sequence_string(s, func=float):
	if type(s) != str:
		return s
	else:
		v = np.array([func(n) for n in re.split(nonnumeric, s) if n != ''])
		return v


def add_dict_to_df(result, df, prefix=None):
	for k, v in result.items():
		if prefix is not None:
			k = prefix + k
		if hasattr(v, 'shape') and len(v.shape) > 1 and v.shape[1] > 1 and v.shape[0] == df.shape[0]:
			df[k] = list(v)
		elif len(v) == df.shape[0]:
			df[k] = v
	return df


def sample_df(df, k, seed=None):
	if k == 'all':
		k = df.shape[0]
	shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
	return shuffled.iloc[:k], shuffled.iloc[k:]


def sample_df_by_column(supplemental_data_df, n=None, seed=None, column=None, prefer_high_values=None, sampling_method=None):
	if sampling_method == 'deterministic':
		sample_df = supplemental_data_df.sort_values(column, ascending=(not prefer_high_values)).iloc[0:n]
	else:
		raise Exception('Unknown sampling method')

	if sample_df.shape[0] < n:  # shouldn't ever happen
		raise Exception(f'Error: only able to sample {sample_df.shape[0]} rows from df instead of desired {n}')

	return sample_df


def subdir_paths(dir_path):
	try:
		filenames = os.listdir(dir_path)
		paths = [(filename, os.path.join(dir_path, filename)) for filename in filenames]
		subdir_paths = [(filename, filepath) for filename, filepath in paths if os.path.isdir(filepath)]
		return sorted(subdir_paths, key=lambda t: t[0])
	except:
		return []


import json


def dump_json(obj=None, path=None, directory=None, filename=None, overwrite_existing=True):
	if path is None:
		path = os.path.join(directory, filename)

	if os.path.exists(path) and not overwrite_existing:
		done = False
		i = 1
		while not done:
			new_path = number_filepath(path, i)
			if os.path.exists(new_path):
				i += 1
			else:
				path = new_path
				done = True

	directory = os.path.dirname(path)
	if not os.path.exists(directory):
		os.makedirs(directory)

	with open(path, 'w') as f:
		json.dump(obj, f, cls=ExpandedEncoder)


from json import JSONEncoder
from abc import ABCMeta


class ExpandedEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, (list, dict, str, bytes, int, float, bool, type(None))):
			return JSONEncoder.default(self, obj)
		elif isinstance(obj, ABCMeta):
			return str(obj)
		else:
			raise Exception(f'Don\'t know how to serialize object of type {type(obj)}')


def number_filepath(path, num):
	filename = os.path.basename(path)
	directory = os.path.dirname(path)
	parts = filename.split('.')
	new_filename = '.'.join([parts[0] + '-' + str(num)] + parts[1:])
	return os.path.join(directory, new_filename)


def load_json(path=None, directory=None, filename=None):
	if path is None:
		path = os.path.join(directory, filename)
	with open(path, 'r') as f:
		r = json.load(f)
	return r


def best_dev_row(eval_df, metric, training_type=None, disallow_initial=False):
	iprint('Selecting best dev set performance row')
	if training_type is not None:
		eval_df = eval_df[eval_df['training_type'] == training_type]

	if disallow_initial:
		eval_df = eval_df[eval_df['step'] > 0]

	dev_df = eval_df[eval_df['set'] == 'dev']
	best_row = dev_df.loc[dev_df[metric].idxmax()]
	iprint(f'Best row was {best_row["training_type"]} step {best_row["step"]}  with {metric} = {best_row[metric]}')

	return best_row


def annotation_to_rationale(annotation_list, tokenization, ttree=None, tokenwise_annotations=False, vector_type=None):
	'''
	Convert a set of spans of characters or tokens (the annotations), to a mask over a token list defined by the tokenization input

	:param annotation_list: A list of tuples of the form (label, start, end) or (start, end)
	:param tokenization: A list of character spans which define the tokenization of some text
	:param ttree: An interval tree, if we are running this function on the same text over and over again
	:param tokenwise_annotation: If this is false, then the annotations in annotation_list are character spans. If true, they are token spans.
	:return:
	'''
	rationale = [0 for span in tokenization]

	if not tokenwise_annotations:
		if not ttree:
			ttree = it.IntervalTree()
			for i, (tstart, tend) in enumerate(tokenization):
				ttree.addi(tstart, tend, data=i)

		for annotation_tuple in annotation_list:
			if len(annotation_tuple) == 3:
				label, astart, aend = annotation_tuple
			else:
				astart, aend = annotation_tuple
			for token in ttree.search(astart, aend):
				rationale[token.data] = 1
	else:
		for annotation_tuple in annotation_list:
			astart, aend = annotation_tuple
			for i in range(astart, aend):
				rationale[i] = 1

	if vector_type:
		rationale = np.array(rationale, dtype=vector_type)

	return rationale


def rationale_to_annotation(rationale, token_spans=None, text=None):
	annotations = []
	r = 0
	last = None
	current = None
	start = None
	end = None

	if rationale is None or np.all(np.isnan(rationale)):
		return []

	if token_spans is None and text is not None:
		t_start = 0
		token_spans = []
		tokens = text.split(' ')
		for token in tokens:
			token_spans.append([t_start, t_start + len(token)])
			t_start += (len(token) + 1)

	while r < len(rationale):
		current = rationale[r]
		if rationale[r] == 1:
			if last == 1:
				end = token_spans[r][1] if token_spans is not None else r + 1
			else:
				start, end = token_spans[r] if token_spans is not None else (r, r + 1)

		else:
			if last == 1:
				annotations.append((start, end))
			else:
				pass

		last = current
		r += 1

	if last == 1:
		annotations.append((start, end))

	return annotations


def replace_wiki_tokens(s):
	'''
	Replace tokens specific to the Wikipedia data with correct characters
	:param s:
	:return:
	'''
	return s.replace('TAB_TOKEN', '\t').replace('NEWLINE_TOKEN', '\n')


def year2datetime(year):
	return datetime(year=year, month=1, day=1)


def infill_spans(spans, max_length, infill_value=0):
	'''
	Given a list of spans, order them and fill in the gaps
	:param spans: list of dictionaries e.g.
	'spans': [{'span': [14, 15], 'value': 0.768506348133087},
	  {'span': [19, 20], 'value': 0.7824859023094171},
	  {'span': [3, 4], 'value': 0.787116527557373},
	  {'span': [13, 14], 'value': 0.7907468676567071},
	  {'span': [16, 17], 'value': 0.790775954723358},
	  {'span': [6, 7], 'value': 0.833489596843719}]}
	:param max_length:
	:param infill_value:
	:return:
	'''

	spans = sorted(spans, key=lambda d: d['span'][0])
	current = 0
	rspans = []

	# fill in space before each existing span
	for span in spans:
		if span['span'][0] > current:
			newspan = {'span': [current, span['span'][0]], 'value': infill_value}
			current = span['span'][1]
			rspans.append(newspan)
		rspans.append(span)

	# Add final infill
	if max_length > rspans[-1]['span'][1]:
		rspans.append({'span': [rspans[-1]['span'][1], max_length], 'value': infill_value})

	return rspans




class IncompatibleHyperparameterException(Exception):
	pass


def set_display_options():
	np.set_printoptions(linewidth=1000, precision=3, suppress=True, threshold=3000)
	torch.set_printoptions(linewidth=1000, precision=3, threshold=3000, sci_mode=False,edgeitems=50)
	pd.set_option('display.width', 3000)
	pd.set_option('display.max_colwidth', 50)
	pd.set_option('display.max_columns', 20)
	pd.set_option('precision', 3)
	pd.options.mode.chained_assignment = None


def ensure_containing_dir(path):
	direc = os.path.dirname(path)
	if not os.path.isdir(direc): os.makedirs(direc)

def ensure_containing_dir_exists(path):
	ensure_containing_dir(path)

def ensure_dir_exists(direc):
	if not os.path.isdir(direc): os.makedirs(direc)


def update_json_file(outdicts: Union[Dict, List[Dict]], filepath: str, reset: bool = False):
	if not type(outdicts) == list:
		outdicts = [outdicts]

	df = pd.DataFrame(outdicts)

	if not reset and os.path.exists(filepath):
		old_df = pd.read_json(filepath, orient='records', lines=True)
		df = pd.concat([old_df, df], axis=0)

	df.to_json(filepath, orient='records', lines=True)


def run_in_new_process(function:Callable, args:Tuple=(), kwargs:Dict={}):
	try:
		p = mp.Process(target=function, args=args, kwargs=kwargs)
		p.start()
		p.join() # this blocks until the process terminates
		p.close()
	except Exception as ex:
		print(ex)
		traceback.print_exc()

# def tokens_to_spans(tokens, sep=' '):
# 	'''
# 	Convert a sequence of tokens to a sequence of spans
# 	:param tokens:
# 	:return:
# 	'''
# 	spans = []
# 	current_index = 0
# 	for token in tokens:
# 		spans.append([current_index, current_index + len(token)])
# 		current_index += len(token) + len(sep)
#
# 	return spans


def tokens_to_spans(tokens:List[str], text:str=None, sep:str=None):
	'''
	Take a set of tokens generated by some unknown tokenizer, and match them to the original raw text, outputting the corresponding span for each token.

	Useful when you have a dataset that provides tokens and original text, but not the spans that allow you to convert between them

	:param tokens:
	:param text:
	:return:
	'''


	spans = []
	cursor = 0
	for i,token in enumerate(tokens):
		try:
			if text is not None:
				token_index = text.index(token, cursor)
			else:
				token_index = cursor + len(sep)

			span = [token_index, token_index+len(token)]
			spans.append(span)
			cursor = token_index+len(token)
		except ValueError as ex:
			print('Could not align tokens with text.')
			print(f'\tToken: "{token}"; Surrounding tokens: {tokens[i-5:i+5]}')
			print(f'\tText: "{text[cursor:cursor+5]}"...; Surrounding text: "{text[cursor-30:cursor+30]}"')

			return None


	return spans

def render_rationale_tensors(token_ids:torch.Tensor,
							 tokenizer,
							 rationale:torch.Tensor=None,
							 padding_mask:torch.Tensor=None,
							 threshold:float=0.5,
							 do_print=False,
							 show_rationale_values=True):


	masked_tokens = tokenizer.convert_ids_to_tokens(token_ids)
	if rationale is not None:
		if show_rationale_values:
			masked_tokens = [f'{token}({zi:.3f})' if zi >= threshold else f'{"_"*len(token)}({zi:.3f})' for token, zi in zip(masked_tokens, rationale)]
		else:
			masked_tokens = [token if zi >= threshold else '_'*len(token) for token, zi in zip(masked_tokens, rationale)]

	if padding_mask is not None:
		masked_tokens = [token for token, pi in zip(masked_tokens, padding_mask) if pi > 0]

	text = ' '.join(masked_tokens)

	if do_print:
		iprint(text)

	return text

def multi_output_apply(df:pd.DataFrame=None, function:Callable = None, columns:List[str] = None):
	'''Doing an apply that creates multiple outputs is sinfully slow, so this just does that with a concat'''

	outputs = []
	for i, row in df.iterrows():
		outputs.append(function(row))

	output_df = pd.DataFrame(outputs,index=df.index,columns=columns)

	return pd.concat([df,output_df],axis=1)
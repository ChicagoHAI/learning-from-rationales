import itertools
import operator
from copy import deepcopy
from typing import Dict, List
from util.print_util import set_print, iprint
from pprint import pprint
import importlib
from util.misc_util import load_json
import json


'''
Various utility functions for manipulation of iterables
'''


def recursive_get_combos(input_dict,
						 exclude_combos:List=None,
						 require_combos:List=None,
						 exclude_keys:List=[],
						 name_keys:List=['root']):
	'''
	Take an arbitrarily nested dictionary of lists and sub-dictionaries, look for all lists contained in the structure, and
	return one copy of the whole structure for each unique combo of those lists (as appropriate given the tree structure).

	Primarily used for expanding a single config dict specifying a bunch of different hyperparameter combinations into
	one config dict per combination that we can use to parameterize a model, trainer, etc.

	Also produces a somewhat-unique combination name for each of the generated dictionaries, for use in output directory naming, etc.

	Example:
	{'dataset': [{'name': 'esnli', 'train': '/home/esnli'},
				  {'name': 'multirc', 'train': '/home/multirc'}],
	 'script': {'output_dir': '/home/somedir', 'trial': [1, 2]}}

	 -->

	 [{'combo_name': 't=1',
	  'dataset': {'name': 'esnli', 'train': '/home/esnli'},
	  'script': {'output_dir': '/home/somedir', 'trial': 1}},
	 {'combo_name': 't=1',
	  'dataset': {'name': 'multirc', 'train': '/home/multirc'},
	  'script': {'output_dir': '/home/somedir', 'trial': 1}},
	 {'combo_name': 't=2',
	  'dataset': {'name': 'esnli', 'train': '/home/esnli'},
	  'script': {'output_dir': '/home/somedir', 'trial': 2}},
	 {'combo_name': 't=2',
	  'dataset': {'name': 'multirc', 'train': '/home/multirc'},
	  'script': {'output_dir': '/home/somedir', 'trial': 2}}]


	:param input: a dictionary of lists, dictionaries and scalar values
	:param exclude_combos:
	:param require_combos:
	:param exclude_keys: any keys of sub-dictionaries that should be resolved as if they were scalars (e.g. "classes" in the dataset configs)
	:param name_keys: any keys that should be given a name. 'root' refers to the root level dictionary(s)
	:return:
	'''
	if require_combos is not None:
		raise NotImplementedError('Required combinations not supported yet')
	if exclude_combos is not None:
		raise NotImplementedError('Excluded combinations not supported yet')

	combo_dicts = recursive_split(input_dict, name_keys=name_keys, exclude_keys=exclude_keys, root=True)

	return combo_dicts


def read_config(config_spec:str):
	if config_spec.endswith('.py'):
		config = importlib.import_module(str).config
	elif config_spec.endswith('.json'):
		config = load_json(path=config_spec)
	elif config_spec.endswith('}'):
		config =  json.loads(config_spec)
	else:
		raise Exception(f"Don't know how to process input as config file: {config_spec}")
	return config


def flatten(container):
	'''
	Taken from https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
	:param container:
	:return:
	'''
	for i in container:
		if isinstance(i, list):
			for j in flatten(i):
				yield j
		else:
			yield i


def recursive_split(obj, name_keys=[], exclude_keys=[], root=False):
	'''
	This function goes through a nested dictionary recursively, and, every time it encounters
	a list, generates a copy of the whole structure containing each value of that list. When
	multiple lists occur in the same subdictionary, it calculates all combinations of values of those lists.
	:param obj: dictionary
	:param track_uniques: whether to track the unique values discovered at each level, to later be accumulated into a name for each combo
	:param exclude_keys: any keys of sub-dictionaries that should be resolved as if they were scalars
	:return:
	'''
	# iprint(f'Input ({level}): {pformat(obj)}')

	if isinstance(obj, dict):
		obj = {key: recursive_split(value,
									name_keys=name_keys,
									exclude_keys=exclude_keys) if not key in exclude_keys else value for key, value in obj.items()}
		# dict_dict ={key:value for key, value in obj.items() if isinstance(value, dict)}
		list_dict = {key: value for key, value in obj.items() if isinstance(value, list) and not key in exclude_keys}
		non_list_dict = {key: value for key, value in obj.items() if not isinstance(value, list) or key in exclude_keys}


		if len(list_dict) > 0:
			split_list_dicts = list(product_dict(**list_dict))

			robj = []
			for split_list_dict in split_list_dicts:
				combined_dict = {**deepcopy(split_list_dict), **deepcopy(non_list_dict)}

				#Collect unique key/value pairs for this particular dict, and accumulate them from children, popping them from children
				#unless childrens' keys are in name_keys
				local_uniques = {key:value for key, value in split_list_dict.items() if is_scalar(value)}
				for child_key, child_value in combined_dict.items():
					if isinstance(child_value, dict) and 'unique' in child_value:
						child_uniques = child_value.pop('unique')
						local_uniques.update(child_uniques)
						if child_key in name_keys:
							child_value['combo_name'] = generate_paramdict_name(child_uniques)

				# If we're at the root, only accumulate into a name if 'root' is in the name_keys list. Otherwise don't accumulate at all
				if root:
					if 'root' in name_keys:
						combined_dict['combo_name'] = generate_paramdict_name(local_uniques)
				else:
					combined_dict['unique'] = local_uniques


				robj.append(combined_dict)

		else:
			robj =  [non_list_dict]


	elif isinstance(obj, list):
		robj = list(flatten(recursive_split(item,
											name_keys=name_keys,
											exclude_keys=exclude_keys) for item in obj))
	else:
		robj = obj

	# iprint(f'Output ({level}): {pformat(robj)}')

	return robj

def is_scalar(val):
	if not isinstance(val, (dict, list)):
		return True
	else:
		return False

def recursive_pop(d:Dict, key:str, vals=[]):

	if isinstance(d,dict):
		if key in d:
			vals.append(d.pop(key))
		for val in d.values():
			recursive_pop(val, key, vals)

	return vals


def get_param_combos(input, exclude_combos=None, require_combos=None, verbose=False, add_name=True):
	'''
	Older function, superseded by recursive_get_combos

	Like itertools.product but for dictionaries. Looks for all values that are lists, and returns a dictionary for each unique combination of these value lists.

	For example, will split
	{'a':[100,200], 'b':300}
	into
	[{'name':'a=100','params':{'a':100,'b':300}},
	{'name':'a=200','params':{'a':200,'b':300}}]

	Purpose is mainly for converting a config file into a set of hyperparameter dictionaries to try

	input can also be a list or tuple of dictionaries, in which case it will be combined into one big dictionary, the combinations
	will be calculated, and then the output dictionaries will be split back into tuples according to the keys of the original inputs
	'''

	if type(input) == dict:
		input_dict = input
	elif type(input) == list or type(input) == tuple:
		input_dict = {}
		for sub_dict in input:
			if sub_dict is not None:
				input_dict.update(sub_dict)

	if not verbose: previous = set_print(False)
	iprint('Splitting params')
	anonymous_constants = {k: v for k, v in input_dict.items() if not type(v) == list}
	list_params = {k: v for k, v in input_dict.items() if type(v) == list}

	outputs = []
	excluded_outputs = []
	required_outputs = []
	param_combos = list(product_dict(**list_params))

	iprint('{} combos'.format(len(param_combos)))

	for unique_param_combo in param_combos:
		name = generate_paramdict_name(unique_param_combo)
		param_combo = {**unique_param_combo, **anonymous_constants}

		if add_name:
			output = {'name': name, 'params': param_combo}
		else:
			output = param_combo
		# Check to see if this param combo needs to be excluded because it violates one of the exclusion or requirement rules
		exclude = False

		if exclude_combos:
			for ed in exclude_combos:
				if dict_contains(param_combo, ed):
					excluded_outputs.append(output)
					exclude = True
					iprint('Violated exclusion {}'.format(ed))

		# Combo params needs to either fully contain or fully not contain every required combo
		if require_combos:
			for req in require_combos:
				num_present = 0
				num_contained = 0
				for rkey, rval in req.items():
					# if the rkey is not in the dict, we consider the dict to
					if rkey in param_combo:
						num_present += 1
						if param_combo[rkey] == rval:
							num_contained += 1
				if num_contained != num_present and num_contained != 0:
					required_outputs.append(output)
					exclude = True
					iprint('Violated required combo {}'.format(req))

		if not exclude:
			outputs.append(output)
			iprint('Including {}'.format(name))
		else:
			iprint('Excluding {}'.format(name))

	if type(input) == list or type(input) == tuple:
		for output in outputs:
			split_params = []
			for sub_input in input:
				if sub_input is not None:
					split_params.append({k: v for k, v in output['params'].items() if k in sub_input})
				else:
					split_params.append(None)
			output['params'] = split_params

	if not verbose: set_print(previous)

	return outputs


def product_dict(**kwargs):
	'''
	Version of itertools.product that operates on a dictionary of lists to return a list of dictionaries of key-value combinations

	e.g. {a:[1,2],b:[3]} --> [{a:1,b:3},{a:2,b:3}]
	:param kwargs:
	:return:
	'''
	keys = kwargs.keys()
	vals = kwargs.values()
	for instance in itertools.product(*vals):
		yield dict(zip(keys, instance))


def deep_dict_eq(_v1, _v2):
	'''
	Deep comparison of two dictionaries. Taken from https://gist.github.com/samuraisam/901117/521ed1ff8937cb43d7fcdbc1a6f6d0ed2c723bae
	:param _v1:
	:param _v2:
	:return:
	'''
	def _deep_dict_eq(d1, d2):
		k1 = sorted(d1.keys())
		k2 = sorted(d2.keys())
		if k1 != k2: # keys should be exactly equal
			return False
		return sum(deep_dict_eq(d1[k], d2[k]) for k in k1) == len(k1)

	def _deep_iter_eq(l1, l2):
		if len(l1) != len(l2):
			return False
		return sum(deep_dict_eq(v1, v2) for v1, v2 in zip(l1, l2)) == len(l1)

	op = operator.eq
	c1, c2 = (_v1, _v2)

	# guard against strings because they are also iterable
	# and will consistently cause a RuntimeError (maximum recursion limit reached)

	if isinstance(_v1, str):
		pass
	else:
		if isinstance(_v1, dict):
			op = _deep_dict_eq
		else:
			try:
				c1, c2 = (list(iter(_v1)), list(iter(_v2)))
			except TypeError:
				c1, c2 = _v1, _v2
			else:
				op = _deep_iter_eq

	return op(c1, c2)


def generate_paramdict_name(paramdict, abbreviate_names=True, delimiter='_', singletons_only=True, default='default'):

	items = paramdict.items()
	if singletons_only:
		items = [item for item in items if not isinstance(item[1],(list, dict))]

	if len(items) == 0:
		return default
	else:
		return join_names_and_values(*zip(*sorted(items, key=lambda t: t[0])), abbreviate_names=abbreviate_names, delimiter=delimiter)


def join_names_and_values(names, values, abbreviate_names=True, delimiter='_'):
	if abbreviate_names:
		combo_name = delimiter.join('{}={}'.format(abbreviate(name), value) for name, value in zip(names, values))
	else:
		combo_name = delimiter.join('{}={}'.format(name, value) for name, value in zip(names, values))
	return combo_name


def abbreviate(s, split_token='_'):
	return ''.join(w[0] for w in s.split(split_token))


def parse_combo_name(combo_name, abb_dict=None):
	'''
	Parse the name of a param combo as created by generate_paramdict_name
	:param combo_name:
	:param abb_dict:
	:return:
	'''
	#     print(combo_name)
	pieces = [piece.split('_') for piece in combo_name.split('=')]
	#     print(pieces)
	valdict = {}
	for i in range(1, len(pieces)):
		#         print(pieces[i-1][-1])
		#         print('_'.join(pieces[i][:-1]))
		key = pieces[i - 1][-1]
		value = '_'.join(pieces[i][:-1]) if i < len(pieces) - 1 else '_'.join(pieces[i])
		value = parse_value(value)

		if abb_dict is not None and key in abb_dict:
			key = abb_dict[key]
		valdict[key] = value

	return valdict

def parse_value(value:str):
	try:
		value = int(value)
	except:
		try:
			value=float(value)
		except:
			if value.lower() in ['true','false']:
				value = value.lower() == 'true'

	return value


def dict_contains(d1, d2):
	'''
	Check to see if dictionary 2 is contained with dictionary 1
	:param d1:
	:param d2:
	:return:
	'''
	contains = True
	for k, v in d2.items():
		if not (k in d1 and d1[k] == v):
			contains = False
			break
	return contains


def combine_dicts(*dicts):
	if all([d is None for d in dicts]):
		return None

	combined = {}
	for d in dicts:
		if d is not None:
			combined.update(d)
	return combined


def combine_lists(*lists):
	if all([l is None for l in lists]):
		return None

	combined = []
	for l in lists:
		if l is not None:
			combined.extend(l)
	return combined
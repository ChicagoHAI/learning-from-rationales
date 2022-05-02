from util.iter_util import recursive_get_combos, product_dict, deep_dict_eq
from copy import deepcopy
from pprint import pprint


def test_recursive_get_combos():
	model_dict, partial_dict, full_dict, split_dicts = define_dicts()

	for i in range(len(split_dicts)):
		for j in range(i + 1, len(split_dicts)):
			assert not deep_dict_eq(split_dicts[i], split_dicts[j])

	t1 = recursive_get_combos(partial_dict)
	pprint(partial_dict)
	print('---------------------------------------------------------------------------------')
	pprint(t1)
	print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	pprint(full_dict)
	t2 = recursive_get_combos(full_dict)
	print('---------------------------------------------------------------------------------')
	pprint(t2)
	pass


def split_dict(adict):
	singleton_dict = {key: value for key, value in adict.items() if not isinstance(value, list)}
	list_dict = {key: value for key, value in adict.items() if isinstance(value, list)}

	split_list_dicts = list(product_dict(**list_dict))

	combined_dicts = [{**split_list_dict, **deepcopy(singleton_dict)} for split_list_dict in split_list_dicts]

	return combined_dicts


def define_dicts():
	model_dict = {'class': 'BertRationaleModel',
				  'masking_strategy': 'removal',
				  'dropout_rate': 0.1,
				  'learning_rate': [2e-5, 2e-6],
				  'sparsity_loss_weight': [0.01, 0.05]}

	partial_dict = {'script': {'output_dir': '/home/somedir',
							   'trial': [1, 2]},
					'datasets': [{'name': 'esnli',
								  'train': '/home/esnli'},
								 {'name': 'multirc',
								  'train': '/home/multirc'}]}

	# partial_dict = {'model_trainer': [{'model': {'class': 'BertRationaleModel',
	# 											 'learning_rate': [2e-5, 2e-6]},
	# 								   'trainer': {'num_epochs': 3,
	# 											   'gpus': 1}},
	# 								  {'model': {'class': 'BertRationaleModel',
	# 											 'learning_rate': 2e-5},
	# 								   'trainer': {'num_epochs': 3,
	# 											   'gpus': [2]}}]}

	# partial_dict = {'model': {'class': 'BertRationaleModel',
	# 											 'learning_rate': [2e-5, 2e-6]},
	# 								   'trainer': {'num_epochs': 3,
	# 											   'gpus': 1}}

	full_dict = {'script': {'output_dir': '/home/somedir',
							'trial': [1, 2]},
				 'datasets': [{'name': 'esnli',
							   'train': '/home/esnli'},
							  {'name': 'multirc',
							   'train': '/home/multirc'}],
				 'model_trainer': [{'model': {'class': 'BertRationaleModel',
											  'learning_rate': [2e-5, 2e-6]},
									'trainer': {'num_epochs': 3,
												'gpus': 1}},
								   {'model': {'class': 'BertRationaleModel',
											  'learning_rate': 2e-5},
									'trainer': {'num_epochs': 3,
												'gpus': [2]}}]}

	#################################################################################

	split_dicts = [
		{'script': {'output_dir': '/home/somedir',  # 1
					'trial': 1},
		 'datasets': {'name': 'esnli',
					  'train': '/home/esnli'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-5},
						   'trainer': {'num_epochs': 3,
									   'gpus': 2}}},

		{'script': {'output_dir': '/home/somedir',  # 2
					'trial': 1},
		 'datasets': {'name': 'esnli',
					  'train': '/home/esnli'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-5},
						   'trainer': {'num_epochs': 3,
									   'gpus': 1}}},

		{'script': {'output_dir': '/home/somedir',  # 3
					'trial': 1},
		 'datasets': {'name': 'esnli',
					  'train': '/home/esnli'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-6},
						   'trainer': {'num_epochs': 3,
									   'gpus': 1}}},

		{'script': {'output_dir': '/home/somedir',  # 4
					'trial': 1},
		 'datasets': {'name': 'multirc',
					  'train': '/home/multirc'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-5},
						   'trainer': {'num_epochs': 3,
									   'gpus': 2}}},

		{'script': {'output_dir': '/home/somedir',  # 5
					'trial': 1},
		 'datasets': {'name': 'multirc',
					  'train': '/home/multirc'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-5},
						   'trainer': {'num_epochs': 3,
									   'gpus': 1}}},

		{'script': {'output_dir': '/home/somedir',  # 6
					'trial': 1},
		 'datasets': {'name': 'multirc',
					  'train': '/home/multirc'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-6},
						   'trainer': {'num_epochs': 3,
									   'gpus': 1}}},

		{'script': {'output_dir': '/home/somedir',  # 7
					'trial': 2},
		 'datasets': {'name': 'esnli',
					  'train': '/home/esnli'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-5},
						   'trainer': {'num_epochs': 3,
									   'gpus': 2}}},

		{'script': {'output_dir': '/home/somedir',  # 8
					'trial': 2},
		 'datasets': {'name': 'esnli',
					  'train': '/home/esnli'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-5},
						   'trainer': {'num_epochs': 3,
									   'gpus': 1}}},

		{'script': {'output_dir': '/home/somedir',  # 9
					'trial': 2},
		 'datasets': {'name': 'esnli',
					  'train': '/home/esnli'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-6},
						   'trainer': {'num_epochs': 3,
									   'gpus': 1}}},

		{'script': {'output_dir': '/home/somedir',  # 10
					'trial': 2},
		 'datasets': {'name': 'multirc',
					  'train': '/home/multirc'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-5},
						   'trainer': {'num_epochs': 3,
									   'gpus': 2}}},

		{'script': {'output_dir': '/home/somedir',  # 11
					'trial': 2},
		 'datasets': {'name': 'multirc',
					  'train': '/home/multirc'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-5},
						   'trainer': {'num_epochs': 3,
									   'gpus': 1}}},

		{'script': {'output_dir': '/home/somedir',  # 12
					'trial': 2},
		 'datasets': {'name': 'multirc',
					  'train': '/home/multirc'},
		 'model_trainer': {'model': {'class': 'BertRationaleModel',
									 'learning_rate': 2e-6},
						   'trainer': {'num_epochs': 3,
									   'gpus': 1}}},
	]

	return model_dict, partial_dict, full_dict, split_dicts


if __name__ == '__main__':
	test_recursive_get_combos()

from typing import Dict
import os

from util.misc_util import ensure_dir_exists
from util.read_data_util import read_dataset
from pytorch_lightning import loggers as pl_loggers
from util.print_util import iprint, add_log_destination, remove_log_destination
from lightning_component.progress_bar import ManualWidthProgressBar
from lightning_component.trainer import ModifiedTrainer
from model.resolve_model_class import resolve_model_class
import multiprocessing as mp

seed = 1234
import pandas as pd
from util.gpu_util import choose_gpus
from pprint import pformat
import torch
from experiment_scripts.sub_scripts.tune_batch_size import tune_batch_size
from experiment_scripts.sub_scripts.test_model import test_model

def evaluate_model(producing_model_dir: str,
				   consuming_model_dir: str,
				   config: Dict,
				   evaluation_dir: str,
				   batch_size_cache:Dict[str,Dict[str,int]]=None,
				   desc:str='',
				   queue: mp.Queue= None):
	'''

	
	:param producing_model_dir: 
	:param consuming_model_dir: 
	:param config: 
	:return: 
	'''
	# paramset_dir = os.path.join(experiment_dir, paramset["name"])

	iprint('Cross-evaluating model!')
	iprint(desc)
	iprint(f'Producing model: {config["model_trainer"]["producing"]["model"]["name"]}; params: {config["model_trainer"]["producing"]["combo_name"]}')
	iprint(f'Consuming model: {config["model_trainer"]["consuming"]["model"]["name"]}; params: {config["model_trainer"]["consuming"]["combo_name"]}')

	# val_output_path = os.path.join(evaluation_dir, 'best_val_output', 'epoch_-1_predictions.json')
	# test_output_path = os.path.join(evaluation_dir, 'test_output', 'epoch_-1_predictions.json')
	# if not config['script']['overwrite_results'] and os.path.exists(val_output_path) and os.path.exists(test_output_path):
	# 	iprint('This evaluation appears to have been done already. So, skipping it.')
	# 	return

	producing_model_predictions_path = os.path.join(producing_model_dir, 'test_output', 'epoch_-1_predictions.json')
	# if not os.path.exists(producing_model_predictions_path):
	# 	iprint(f'ERROR: No producing model predictions found at {producing_model_predictions_path}. Skipping this evaluation.')
	# 	return

	consuming_model_path = os.path.join(consuming_model_dir, 'checkpoints', 'best.ckpt')
	# if not os.path.exists(consuming_model_path):
	# 	iprint(f'ERROR: No consuming model checkpoint found at {consuming_model_path}. Skipping this evaluation.')
	# 	return


	ensure_dir_exists(evaluation_dir)
	add_log_destination(evaluation_dir)

	print('#' * 100)
	iprint('Evaluating model!')
	iprint(f'Saving results in {evaluation_dir}')
	iprint('Full configuaration for this evaluation:')
	iprint(pformat(config))

	producing_model_config = config['model_trainer']['producing']['model']
	consuming_model_config = config['model_trainer']['consuming']['model']

	dataset_config = config['dataset']
	read_dataset_config = config['read_dataset']
	script_config = config['script']
	# trainer_config = config['model_trainer']['trainer']
	consuming_trainer_config = config['model_trainer']['consuming']['trainer']
	consuming_trainer_params = consuming_trainer_config['params']




	if script_config.get('gpus') != 'manual':
		gpus = choose_gpus(verbose=True, num=script_config.get('gpus'), max_utilization=21000, align_with_cuda_visible=True)
	else:
		gpus = 1

	producing_model_name = producing_model_config['name']
	# Initialize model
	consuming_model_params = consuming_model_config['params']
	consuming_model_class = resolve_model_class(consuming_model_config['class'])
	consuming_model = consuming_model_class(**consuming_model_params,
											num_classes=len(dataset_config['classes']),
											output_dir=evaluation_dir)
	adjust_model_settings_for_crossevaluation(consuming_model, config)

	consuming_model_name = consuming_model_config['name']

	# Read datasets
	# train_df, train_dataset = read_dataset(
	# 	dataset_config['train'],
	# 	consuming_model.tokenizer,
	# 	dataset_config['classes'],
	# 	**read_dataset_config)

	val_df, val_dataset = read_dataset(
		dataset_config['dev'],
		consuming_model.tokenizer,
		dataset_config['classes'],
		**read_dataset_config)

	test_df, test_dataset = read_dataset(
		dataset_config['test'],
		consuming_model.tokenizer,
		dataset_config['classes'],
		**read_dataset_config)

	# Initialize trainer
	logger = pl_loggers.TensorBoardLogger(save_dir=evaluation_dir,
										  version=0,
										  name='logs')
	bar = ManualWidthProgressBar(width=script_config.get('progress_bar_width'))

	# profiler = AdvancedProfiler()
	consuming_trainer = ModifiedTrainer(
		gpus=gpus,
		default_root_dir=evaluation_dir,
		callbacks=[bar],
		logger=logger,
		# profiler=profiler,
		# auto_scale_test_only=True,
		**consuming_trainer_params)

	consuming_trainer.testing = True

	iprint(f"Setting default batch size to {script_config['default_batch_size']}")
	consuming_model.batch_size = script_config['default_batch_size']



	# don't bother running tuning if it isn't going to do anything
	# if consuming_trainer_params.get('auto_scale_batch_size') in ['binsearch', 'power']:
	cache_hit = False
	if batch_size_cache is not None:
		if dataset_config['name'] in batch_size_cache:
			if consuming_model_config['name'] in batch_size_cache[dataset_config['name']]:
				consuming_model.batch_size = batch_size_cache[dataset_config['name']][consuming_model_config['name']]
				cache_hit = True
				iprint(f"Found batch size cache hit for dataset {dataset_config['name']} and model {consuming_model_config['name']}: {consuming_model.batch_size}")
		else:
			batch_size_cache[dataset_config['name']] = {}


	if not cache_hit:
		tune_batch_size(model=consuming_model,
						trainer=consuming_trainer,
						test_dataset=test_dataset,
						model_config=consuming_model_config,
						trainer_config=consuming_trainer_config,
						dataset_config=dataset_config,
						read_dataset_config=read_dataset_config,
						script_config=script_config,
						in_new_process=script_config['tune_batch_size_in_new_process'],
						config=config,
						model_adjustment_function=adjust_model_settings_for_crossevaluation,
						test_only=True)

		if batch_size_cache is not None:
			iprint(f"Adding batch size {consuming_model.batch_size} to cache for dataset {dataset_config['name']} and consuming model {consuming_model_config['name']}")
			batch_size_cache[dataset_config['name']][consuming_model_config['name']] = consuming_model.batch_size
			if queue is not None:
				queue.put(batch_size_cache)


			# if script_config.get('val_check_interval'):
	# 	val_check_interval = min(script_config['val_check_interval'], len(dev_dataset)/model.batch_size)
	# 	trainer.val_check_interval = val_check_interval

	# Train model
	# consuming_model.initialize_dataloaders(train_dataset, val_dataset, max_only=True)




	iprint(f"Loading best checkpoint based on val_loss from \n {consuming_model_path}")
	batch_size = consuming_model.batch_size
	consuming_model = consuming_model_class.load_from_checkpoint(consuming_model_path,
																 **consuming_model_params,
																 num_classes=len(dataset_config['classes']),
																 output_dir=evaluation_dir)
	adjust_model_settings_for_crossevaluation(consuming_model, config)

	iprint(f'Setting model batch size from loaded {consuming_model.batch_size} to tuned {batch_size}')
	consuming_model.batch_size = batch_size


	iprint(f'Running test functions for {desc}')
	if script_config['evaluate_with_val_set']:
		iprint(f'Doing full evaluation of {producing_model_name} rationales WRT {consuming_model_name} model on validation set')
		producing_val_output_df = pd.read_json(os.path.join(producing_model_dir, 'best_val_output', f'epoch_-1_predictions.json'), lines=True, orient='records')
		add_p_alphas_from_df_to_dataset(producing_val_output_df, val_dataset, '/' + producing_model_name)
		assert (all([len(r1) == len(r2) for r1, r2 in zip(producing_val_output_df['rationale'], val_dataset.rationale)]))
		test_model(
			model=consuming_model,
			trainer=consuming_trainer,
			output_prefix='best_val',
			dataset = val_dataset,
			output_directory=evaluation_dir,
			data_df=val_df,
			output_combined_csv=False,
			output_html_sample=False,
			overwrite_existing_output=script_config['overwrite_existing_output']
		)


	if script_config['evaluate_with_test_set']:
		iprint(f'Doing full evaluation of {producing_model_name} rationales WRT {consuming_model_name} model on test set')
		producing_test_output_df = pd.read_json(os.path.join(producing_model_dir, 'test_output', f'epoch_-1_predictions.json'), lines=True, orient='records')
		add_p_alphas_from_df_to_dataset(producing_test_output_df, test_dataset, '/' + producing_model_name)
		test_model(
			model=consuming_model,
			trainer=consuming_trainer,
			output_prefix='test',
			dataset = test_dataset,
			output_directory=evaluation_dir,
			data_df=test_df,
			output_combined_csv=False,
			output_html_sample=False,
			overwrite_existing_output=script_config['overwrite_existing_output']
		)




	iprint(f'Done evaluating! Results located in {evaluation_dir}')
	remove_log_destination(evaluation_dir)



	return


def adjust_model_settings_for_crossevaluation(model, config):
	'''
	Adjust behavior of model depending on circumstances
	:param config:
	:return:
	'''
	#So that the model doesn't evaluate its own rationales (which would be redundant), and we only do edge case rationales upon self-evaluation
	iprint('Adjusting model settings for cross-evaluation.')
	model.suppress_own_rationales = True #So that the model doesn't evaluate its own rationales (which would be redundant)

	if config['model_trainer']['producing']['model']['name'] != config['model_trainer']['consuming']['model']['name']:
		iprint('Not doing edge-case rationales or human rationales')
		model.suppress_human_rationales=True
		model.test_edge_case_rationales = False

	iprint(f"Setting input masking strategy to {config['script']['evaluation_input_masking_strategy']}")
	model.input_masking_strategy = config['script']['evaluation_input_masking_strategy']
	if config['script']['no_rationale_evaluation']:
		iprint('Not evaluating rationales')
		model.never_evaluate_rationales = True

	if config['script']['test_random_rationales']:
		model.test_random_rationales=True
		model.random_rationale_probabilities = config['script']['random_rationale_probabilities']

def add_p_alphas_from_df_to_dataset(df, dataset, suffix):
	assert df.shape[0] == len(dataset), 'Dataset and dataframe lengths do not match.'
	for column in df.columns:
		if column.startswith('p_alpha'):
			dataset.p_alphas[column + suffix] = [torch.tensor(p) for p in df[column]]

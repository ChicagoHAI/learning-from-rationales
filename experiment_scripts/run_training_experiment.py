import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1' #To debug device-side errors
from util.gpu_util import get_gpu_usage

# choose_and_set_available_gpus(max_utilization=15000)

# from config.run_experiment.debug import config as default_config
from util.print_util import iprint, initialize_logging
from util.misc_util import set_display_options, dump_json, run_in_new_process, ensure_dir_exists
from util.config_util import recursive_get_combos

from typing import Dict

import importlib
import multiprocessing as mp

import pytorch_lightning as pl

from experiment_scripts.sub_scripts.train_model import train_model
from pprint import pformat
import argparse
from copy import deepcopy
from config.datasets import datasets
import random

'''
Test 
'''


def main():
	args = read_args()

	iprint(f'Using config {args.config}')
	config = importlib.import_module(args.config).config

	if (config.get('dataset') is None or len(config.get('dataset')) == 0) and args.dataset is not None:
		config['dataset'] = [datasets[args.dataset]]
	print(pformat(config))

	# todo put this somewhere else
	pl.seed_everything(config['script'].get("seed"))
	# choose_and_set_available_gpus(max_utilization=15000)
	run_experiment(config)


def run_experiment(experiment_config: Dict):
	'''
	Run a learning-from-explanation experiment
	:param experiment_config:
	:param output_dir:
	:param train_models:
	:param evaluate_models:
	:return:
	'''
	mp.set_start_method('spawn')
	set_display_options()
	output_dir = experiment_config['script']['output_dir']
	ensure_dir_exists(output_dir)
	# initialize_logging(output_dir,logger_name='lightning')
	iprint(f'Running experiment. Writing global results to {output_dir}')
	dump_json(experiment_config, directory=output_dir, filename='config.json')


	get_gpu_usage()

	configs = recursive_get_combos(experiment_config, exclude_keys=['classes', 'additional_test', 'additional', 'pretrain'], name_keys=['model_trainer'])
	if experiment_config['script']['shuffle_configs']:
		random.shuffle(configs)

	iprint(f'{len(configs)} config combinations found.')


	for config_num, config in enumerate(configs):
		desc = f"config {config_num + 1} of {len(configs)}: {config['dataset']['name']} - {config['model_trainer']['model']['name']} - {config['model_trainer']['trainer']['name']} - {config['model_trainer']['combo_name']}"
		iprint(f'Starting {desc}')
		train_model_within_experiment(output_dir, config)

		iprint(f'Finished with {desc}')

	iprint('Terminus est...')
	return


def train_model_within_experiment(output_dir: str, config: Dict):
	if 'pretrain' in config['model_trainer']:
		config['model_trainer']['combo_name'] += '_p=True'
		pretrain_model_configs = config['model_trainer']['pretrain']
		if type(pretrain_model_configs) == dict:
			pretrain_model_configs= [pretrain_model_configs]

		iprint(f"Pretraining {len(pretrain_model_configs)} models in preparation for training primary model\nPrimary model: {config['model_trainer']['combo_name']}")
		pretrain_directories = []
		for i, pretrain_model_config in enumerate(pretrain_model_configs):

			pretrain_config = deepcopy(config)
			pretrain_config['model_trainer'] = pretrain_model_config
			pretrain_config['script']['load_existing_model'] = True  # So that we don't pretrain the same model over and over
			pretrain_config['script']['is_pretraining'] = True
			# pretrain_config['script']['no_rationale_evaluation'] = True
			pretrain_config['script']['overwrite_existing_output'] = False
			# pretrain_config['model_trainer']['model']['never_evaluate_rationales'] = True  # so that we don't do any evaluation of rationales for the pretrained models

			pretrain_config = recursive_get_combos(pretrain_config, exclude_keys=['classes', 'additional_test', 'additional','pretrain_only', 'pretrain'], name_keys=['model_trainer'])[0]  # to resolve list-valued hyperparameters and generate a combo name

			if pretrain_model_config.get('pretrained_model_pretrained') or 'pretrain' in pretrain_config['model_trainer']: #if the pretrained model was itself pretrained, append the indicator
				#todo I should rethink this naming convention at some point
				pretrain_config['model_trainer']['combo_name'] += '_p=True'

			iprint(f"Pretrain config {i+1} of {len(pretrain_model_configs)} for primary model \nPretrained model: {pretrain_config['model_trainer']['combo_name']}")


			pretrain_dir = os.path.join(output_dir,
										pretrain_config['dataset']['name'],
										# 'pretrain',
										pretrain_config['model_trainer']['model']['name'],
										pretrain_config['model_trainer']['trainer']['name'],
										pretrain_config['model_trainer']['combo_name'])



			iprint(f'Pretraining model {pretrain_config["model_trainer"]["combo_name"]} into {pretrain_dir}')
			pretrain_directories.append(pretrain_dir)
			if not config['script']['train_model_in_new_process']:
				iprint('Pretraining model in this process')
				train_model(pretrain_dir, pretrain_config)
			else:
				iprint('Pretraining model in new process')
				run_in_new_process(train_model, kwargs={'train_directory': pretrain_dir,
														'config': pretrain_config,
														'experiment_directory': output_dir})
	else:
		pretrain_directories = None

	config_dir = os.path.join(output_dir,
							  config['dataset']['name'],
							  config['model_trainer']['model']['name'],
							  config['model_trainer']['trainer']['name'],
							  config['model_trainer']['combo_name'])
	dump_json(config, directory=config_dir, filename='config.json')

	if not config['script']['train_model_in_new_process']:
		iprint('Training model in this process')
		train_model(config_dir, config, pretrain_directories=pretrain_directories, experiment_directory=output_dir)
	else:
		iprint('Training model in new process')
		run_in_new_process(train_model, kwargs={'train_directory': config_dir,
												'config': config,
												'pretrain_directories': pretrain_directories,
												'experiment_directory': output_dir})

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		default='config.run_training_experiment.multirc_example',
		type=str,
		required=False,
		help="Basic experiment config path",
	)

	parser.add_argument(
		"--dataset",
		default=None,
		type=str,
		required=False,
		help="dataset to train on",
	)

	args = parser.parse_args()
	return args

	# Evaluate model

if __name__ == '__main__':
	main()

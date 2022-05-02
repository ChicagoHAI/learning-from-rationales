from typing import Dict
from util.misc_util import ensure_dir_exists
from util.read_data_util import read_dataset
from util.print_util import iprint, add_log_destination, remove_log_destination
from lightning_component.trainer import ModifiedTrainer
from model.resolve_model_class import resolve_model_class
from multiprocessing import Queue
import traceback
from util.gpu_util import choose_gpus

'''
Figure out a good batch size for a given model/trainer/dataset combination using pytorch lightning tuning

Implemented in a multiprocessing-friendly way because I've found this functionality has issues with memory leaks. 
'''


def tune_trainer(model_directory: str,
				 config: Dict,
				 queue: Queue = None,
				 include_train: bool = True,
				 include_val: bool = True,
				 include_test: bool = False,
				 model_trainer_config:Dict=None,
				 ):
	ensure_dir_exists(model_directory)
	add_log_destination(model_directory)
	iprint('Tuning model')

	if model_trainer_config is None:
		model_trainer_config = config['model_trainer']

	model_config = model_trainer_config['model']
	dataset_config = config['dataset']
	read_dataset_config = config['read_dataset']
	script_config = config['script']
	trainer_config = model_trainer_config['trainer']
	model_params = model_config['params']
	trainer_params = trainer_config['params']

	if script_config.get('gpus') != 'manual':
		gpus = choose_gpus(verbose=True, num=script_config.get('gpus'), max_utilization=21000, align_with_cuda_visible=True)
	else:
		gpus = 1

	# Initialize model
	model = resolve_model_class(model_config['class'])(**model_params,
													   num_classes=len(dataset_config['classes']),
													   output_dir=model_directory)

	# Read datasets
	if include_train:
		train_df, train_dataset = read_dataset(
			dataset_config['train'],
			model.tokenizer,
			dataset_config['classes'],
			**read_dataset_config)
	else:
		train_dataset = None

	if include_val:
		val_df, val_dataset = read_dataset(
			dataset_config['dev'],
			model.tokenizer,
			dataset_config['classes'],
			**read_dataset_config)
	else:
		val_dataset = None

	if include_test:
		test_df, test_dataset = read_dataset(
			dataset_config['test'],
			model.tokenizer,
			dataset_config['classes'],
			**read_dataset_config)
	else:
		test_dataset = None

	trainer = ModifiedTrainer(
		gpus=gpus,
		default_root_dir=model_directory,
		**trainer_params)

	model.initialize_dataloaders(train_dataset=train_dataset,
								 val_dataset=val_dataset,
								 test_dataset=test_dataset,
								 max_only=True)
	try:
		trainer.tune(model)
		batch_size = model.batch_size
		iprint(f'Done tuning model. Returning batch size {batch_size}')
	except Exception:
		traceback.print_exc()
		batch_size = model.batch_size
		iprint(f'Exception occurred, so returning pre-existing batch size {batch_size}')

	batch_size = model.batch_size
	remove_log_destination(model_directory)

	if queue is not None:
		queue.put(batch_size)
	return batch_size

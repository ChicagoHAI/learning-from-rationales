from model.old.base_model import BaseModel
from lightning_component.trainer import ModifiedTrainer
from util.dataset import RationaleDataset
from typing import Dict, Callable
from util.print_util import iprint, initialize_logging
import multiprocessing as mp
from util.read_data_util import read_dataset
from model.resolve_model_class import resolve_model_class
from util.gpu_util import choose_gpus
from typing import List

def tune_batch_size(
		model: BaseModel = None,
		trainer: ModifiedTrainer = None,
		train_dataset: RationaleDataset = None,
		val_dataset: RationaleDataset = None,
		test_dataset: RationaleDataset = None,
		model_config: Dict = None,
		trainer_config: Dict = None,
		dataset_config: Dict = None,
		script_config: Dict = None,
		read_dataset_config: Dict = None,
		model_directory: str = None,
		return_queue: mp.Queue = None,
		load_from_configs: bool = False,
		in_new_process: bool = False,
		test_only: bool = False,
		config: Dict = None,
		model_adjustment_function: Callable = None,
		log_directories:List=None
):
	'''
	Tunes the batch size for a given trainer, model and dataset

	If in_new_process is set to true, it forks off a new process and recreates/reloads the trainer, model and dataset. This helps
	avoid memory leaks, which are sometimes an issue when running this tuning process

	If test_mode is set to true, then the trainer is set to trainer.testing = True, which causes it to only run the .test method
	when .fit is called.
	:return:
	'''
	logger = initialize_logging(directories=log_directories,logger_name='lightning')

	if not trainer_config['params'].get('auto_scale_batch_size') in ['binsearch', 'power']:
		iprint(f"No tuning method specified, so returning default batch size {script_config['default_batch_size']}")
		model.batch_size = script_config['default_batch_size']
		return script_config['default_batch_size']

	if load_from_configs:
		model = resolve_model_class(model_config['class'])(**model_config['params'],
														   num_classes=len(dataset_config['classes']))

		if model_adjustment_function is not None:
			model_adjustment_function(model, config)


		if not test_only:
			train_df, train_dataset = read_dataset(
				dataset_config['train'],
				model.tokenizer,
				dataset_config['classes'],
				**read_dataset_config)

			val_df, val_dataset = read_dataset(
				dataset_config['dev'],
				model.tokenizer,
				dataset_config['classes'],
				**read_dataset_config)
		else:
			test_df, test_dataset = read_dataset(
				dataset_config['test'],
				model.tokenizer,
				dataset_config['classes'],
				**read_dataset_config)

		if script_config.get('gpus') != 'manual':
			gpus = choose_gpus(verbose=False, num=script_config.get('gpus'), max_utilization=21000, align_with_cuda_visible=True)
		else:
			gpus = 1

		trainer = ModifiedTrainer(
			gpus=gpus,
			default_root_dir=model_directory,
			logger=logger,
			**trainer_config['params'])

	if not in_new_process:
		iprint('Tuning batch size in this process')
		assert model is not None
		assert trainer is not None
		if test_only:
			iprint('Tuning batch size for testing only')
			assert test_dataset is not None
			model.initialize_dataloaders(test_dataset=test_dataset, max_only=True)
			trainer.testing = True
		else:
			iprint('Tuning batch size for training.')
			assert train_dataset is not None and val_dataset is not None
			model.initialize_dataloaders(train_dataset=train_dataset, val_dataset=val_dataset, max_only=True)



		model.no_file_io = True
		model.no_evaluation_output=True
		test_edge_case_rationales = model.test_edge_case_rationales #so that there are some rationales to evaluate even if we haven't loaded any in from the consuming model
		model.test_edge_case_rationales = True
		# try:
		trainer.tune(model)
		batch_size = model.batch_size
		# except Exception as ex:
		# 	print_exc()
		# 	iprint(f"Returning default batch size {script_config['default_batch_size']}")
		#
		# 	model.batch_size = batch_size = script_config['default_batch_size']

		model.no_file_io = False
		model.no_evaluation_output = False
		model.test_edge_case_rationales = test_edge_case_rationales
		model.clear_dataloaders()


	else:
		iprint('Tuning batch size in new process')
		queue = mp.Queue()
		p = mp.Process(target=tune_batch_size, kwargs={
			'model_directory': model_directory,
			'model_config': model_config,
			'trainer_config': trainer_config,
			'dataset_config': dataset_config,
			'script_config': script_config,
			'read_dataset_config': read_dataset_config,
			'test_only':test_only,
			'load_from_configs': True,
			'in_new_process': False,
			'config':config,
			'model_adjustment_function':model_adjustment_function,
			'return_queue': queue,
			'log_directories':log_directories
		})
		p.start()
		batch_size = queue.get()
		p.join()  # this blocks until the process terminates
		p.close()
		model.batch_size = batch_size
		iprint(f'Setting model batch size to {batch_size}')

	if return_queue is not None:
		return_queue.put(batch_size)
		return
	else:
		return batch_size

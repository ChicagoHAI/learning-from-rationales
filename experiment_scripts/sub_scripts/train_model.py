from typing import Dict, List
import os

from experiment_scripts.sub_scripts.test_model import test_model
from util.misc_util import ensure_dir_exists, dump_json
from util.read_data_util import read_dataset
from lightning_component.checkpoint import NoVersionCheckpoint
import pytorch_lightning as pl
from util.print_util import iprint, remove_log_destination, initialize_logging
from lightning_component.progress_bar import ManualWidthProgressBar
from lightning_component.stopper import VerboseEarlyStopping
from lightning_component.trainer import ModifiedTrainer
from model.resolve_model_class import resolve_model_class
from lightning_component.tensorboard_logger import ModifiedTensorboardLogger


seed = 1234
from util.gpu_util import choose_gpus
from experiment_scripts.sub_scripts.tune_batch_size import tune_batch_size
from config.classes.script_config import TrainingScriptConfig


def train_model(train_directory: str, config: Dict, pretrain_directories: List[str] = None, experiment_directory=None):
	# paramset_dir = os.path.join(experiment_dir, paramset["name"])

	pl.seed_everything(config['script'].get("seed"))
	logger = initialize_logging(directories=[experiment_directory],logger_name='lightning')

	print('#' * 100)
	if config['script']['is_pretraining']:
		iprint('Pretraining model!')
	else:
		iprint('Training model!')
	# iprint('Using following configuration options:')
	# iprint(pformat(config))

	# if 'pretrain' in config['model_trainer'] and pretrain_directories is not None and 'pretrain_only' in config['model_trainer']['pretrain']:
	# 	config['script']['do_training'] = False
	# 	best_checkpoint_path = os.path.join(pretrain_directory, 'checkpoints', 'best.ckpt')
	# 	iprint(f'"pretrain_only" is enabled, so we are disabling training and evaluating the model at \n{best_checkpoint_path}\ninto this directory:\n{train_directory}')
	# else:
	best_checkpoint_path = os.path.join(train_directory, 'checkpoints', 'best.ckpt')

	model_config = config['model_trainer']['model']
	dataset_config = config['dataset']
	read_dataset_config = config['read_dataset']
	script_config = TrainingScriptConfig(**config['script'])
	trainer_config = config['model_trainer']['trainer']
	model_params = model_config['params']
	if 'model_params' in config['model_trainer']:
		model_params.update(config['model_trainer']['model_params'])

	trainer_params = trainer_config['params']
	if 'trainer_params' in config['model_trainer']:
		trainer_params.update(config['model_trainer']['trainer_params'])

	#Check for existing outputs to see if we need to proceed to the testing stage at all
	#The test function will check all this stuff again on a set-by-set basis
	model_exists = os.path.exists(best_checkpoint_path)
	
	best_val_output_path = os.path.join(train_directory, 'best_val_output', 'best_val_epoch_eval.json')
	best_val_output_exists = os.path.exists(best_val_output_path)

	model_has_been_trained = model_exists and best_val_output_exists
	model_should_be_trained =script_config['do_training'] and not (model_has_been_trained and script_config['load_existing_model'])

	test_output_path = os.path.join(train_directory, 'test_output', 'test_epoch_eval.json')
	test_output_exists = os.path.exists(test_output_path)

	train_output_path = os.path.join(train_directory, 'train_output', 'train_epoch_eval.json')
	train_output_exists = os.path.exists(train_output_path)

	all_additional_output_exists = True
	if 'additional_test' in dataset_config:
		for additional_dataset_config in dataset_config['additional_test']:
			iprint(f'Doing additional test on {additional_dataset_config["name"]} ')
			additional_name = additional_dataset_config["name"]
			additional_output_path = os.path.join(train_directory, f'{additional_name}_output', f'{additional_name}_epoch_eval.json')
			all_additional_output_exists = all_additional_output_exists and os.path.exists(additional_output_path)


	model_should_be_tested_on_test = script_config['do_testing'] and not (test_output_exists and not script_config['overwrite_existing_output'])
	model_should_be_tested_on_train = script_config['do_testing']  and script_config['test_on_train'] and not (train_output_exists and not script_config['overwrite_existing_output'])
	model_should_be_tested_on_additional = script_config['do_testing'] and not (all_additional_output_exists and not script_config['overwrite_existing_output'])

	model_should_be_tested = model_should_be_tested_on_test or model_should_be_tested_on_train or model_should_be_tested_on_additional


	#Decide whether to proceed
	if not model_should_be_trained and not model_should_be_tested:
		iprint('The model has already trained, and all test output has already been generated. So nothing further to do.')
		return



	if script_config['ignore_nonexisting_model'] and not model_exists:
		iprint('This model does not appear to have been trained, but we are not training new models. So returning.')
		return


	ensure_dir_exists(train_directory)
	initialize_logging(train_directory,logger_name='lightning')
	# add_log_destination(train_directory)
	dump_json(config, directory=train_directory, filename='config.json')

	if 'comment' in model_config and model_config['comment'] != '' and model_config['comment'] is not None:
		with open(os.path.join(train_directory, 'comment.txt'), 'w') as cf:
			cf.write(model_config['comment'])


	if script_config.get('gpus') != 'manual':
		gpus = choose_gpus(verbose=True, num=script_config.get('gpus'), max_utilization=23000, align_with_cuda_visible=True)
	else:
		gpus = 1


	# Initialize model
	model_class = resolve_model_class(model_config['class'])
	model = model_class(**model_params,
						num_classes=len(dataset_config['classes']),
						output_dir=train_directory)

	# Read datasets
	train_df, train_dataset = read_dataset(
		dataset_config['train'],
		model.tokenizer,
		dataset_config['classes'],
		# add_pseudoexamples=model.add_pseudoexamples,
		# pseudoexample_type=model.pseudoexample_type,
		# pseudoexample_proportion=model.pseudoexample_proportion,
		# pseudoexample_parameter=model.pseudoexample_parameter,
		**read_dataset_config)

	dev_df, dev_dataset = read_dataset(
		dataset_config['dev'],
		model.tokenizer,
		dataset_config['classes'],
		**read_dataset_config)

	test_df, test_dataset = read_dataset(
		dataset_config['test'],
		model.tokenizer,
		dataset_config['classes'],
		**read_dataset_config)

	# model.example_input_array =

	# Initialize trainer
	early_stopper = VerboseEarlyStopping(monitor='val_loss', min_delta=0.001, patience=script_config['patience'], verbose=True)
	checkpoint_params = script_config.get("checkpoint_params", {'monitor': 'val_loss', 'mode': 'min', 'save_last': False})
	checkpoint_callback = NoVersionCheckpoint(verbose=True, filepath=os.path.join(train_directory, 'checkpoints', 'best'), **checkpoint_params)
	pl_logger = ModifiedTensorboardLogger(save_dir=train_directory,
											 log_graph=script_config.get('log_graph'),
										  # version=0,
										  name='logs')

	bar = ManualWidthProgressBar(width=script_config.get('progress_bar_width'))

	# if script_config.get('log_graph'):
	# 	iprint('Logging the graph for the training step using the first training batch')
	# 	first_training_batch = next(iter(model.train_dataloader()))



	trainer = ModifiedTrainer(
		gpus=gpus,
		default_root_dir=train_directory,
		checkpoint_callback=checkpoint_callback,
		callbacks=[bar, early_stopper],
		logger=pl_logger,
		**trainer_params)


	if 'default_batch_size' in script_config:
		iprint(f"Manually setting batch size to {script_config['default_batch_size']}")
		batch_size = model.batch_size = script_config['default_batch_size']
	else:
		batch_size = model.batch_size

		# model.batch_size = dataset_config['batch_size'] #default batch size, may be overrided later

	tune_batch_size(model=model,
					trainer=trainer,
					train_dataset=train_dataset,
					val_dataset=dev_dataset,
					model_config=model_config,
					trainer_config=trainer_config,
					dataset_config=dataset_config,
					read_dataset_config=read_dataset_config,
					script_config=script_config,
					in_new_process=script_config['tune_batch_size_in_new_process'],
					test_only=not script_config['do_training'],
					model_directory=train_directory,
					log_directories=[experiment_directory, train_directory])



	# sort out what pretraining we need to do
	if pretrain_directories is None: pretrain_directories = []
	pretrain_configs = config['model_trainer'].get('pretrain')
	if type(pretrain_configs) == dict:
		pretrain_configs = [pretrain_configs]
	if pretrain_configs is None: pretrain_configs = []
	assert len(pretrain_configs) == len(pretrain_directories)
	pretrain_only = any([pretrain_config.get('pretrain_only') for pretrain_config in pretrain_configs])


	if script_config.get('load_existing_model') and model_exists:
		iprint(f'This model appears to already be trained at {best_checkpoint_path}, so skipping training.')
		model = model_class.load_from_checkpoint(best_checkpoint_path, **model_params,
												 num_classes=len(dataset_config['classes']),
												 strict=False,
												 output_dir=train_directory)



		# return
	else:

		if len(pretrain_directories) > 0:

			for pretrain_directory, pretrain_config in zip(pretrain_directories, pretrain_configs):
				pretrained_checkpoint_path = os.path.join(pretrain_directory, 'checkpoints', 'best.ckpt')
				iprint(f'Loading pretrained parameters from {pretrained_checkpoint_path}')

				# pretrain_config = config['model_trainer']['pretrain']
				pretrain_model_config = pretrain_config['model']
				pretrain_model_params = pretrain_model_config['params']
				if 'model_params' in pretrain_config:
					pretrain_model_params.update(pretrain_config['model_params'])
				pretrained_model = resolve_model_class(pretrain_model_config['class']).load_from_checkpoint(pretrained_checkpoint_path,
																											num_classes=len(dataset_config['classes']),
																											output_dir=None,
																											strict=False,
																											**pretrain_model_params)

				model.load_from_pretrained(pretrained_model, load_to=pretrain_config['load_to'])

		if script_config['do_training'] and not (len(pretrain_directories) > 0 and pretrain_only):
			model.initialize_dataloaders(train_dataset, dev_dataset, max_only=False)

			# example_input_dict = next_functions(iter( model.train_dataloader()))
			# model.example_input_array  =tuple([example_input_dict.get(key) for key in model.forward_args()])
			trainer.fit(model=model)
			iprint(f"Loading best checkpoint based on val_loss from \n {best_checkpoint_path}")
			model = model_class.load_from_checkpoint(best_checkpoint_path, **model_params,
													 num_classes=len(dataset_config['classes']),
													 strict=False,
													 output_dir=train_directory)
		elif not script_config['do_training']:
			iprint('"do_training" set to False, so skipping training (but not pretraining)')
		elif pretrain_only:
			iprint('"pretrain_only" specified in pretraining parameters, so skipping model training.')


	model.batch_size = batch_size


	if script_config.get('no_rationale_evaluation'):
		iprint('Not doing any rationale evaluation')
		model.never_evaluate_rationales = True

	test_model(model=model,
			   trainer=trainer,
			   output_directory=train_directory,
			   output_prefix='best_val',
			   dataset=dev_dataset,
			   data_df=dev_df,
			   output_html_sample=True,
			   overwrite_existing_output= script_config.get('overwrite_existing_output'))


	if script_config['do_testing']:
		test_model(model=model,
				   trainer=trainer,
				   output_directory=train_directory,
				   output_prefix='test',
				   dataset=test_dataset,
				   data_df=test_df,
				   output_html_sample=True,
				   output_combined_csv=script_config.get('output_combined_csv'),
				   overwrite_existing_output= script_config.get('overwrite_existing_output'))

		if script_config['test_on_train']:
			test_model(model=model,
					   trainer=trainer,
					   output_directory=train_directory,
					   output_prefix='train',
					   dataset=train_dataset,
					   data_df=train_df,
					   output_html_sample=True,
					   output_combined_csv=script_config.get('output_combined_csv'),
					   overwrite_existing_output= script_config.get('overwrite_existing_output'),
					   )

		if script_config['test_on_train_sample']:
			train_dataset_sample= train_dataset.sample(n=min(test_df.shape[0], train_df.shape[0]), random_state=seed)
			train_df_sample = train_df.sample(n=min(test_df.shape[0], train_df.shape[0]), random_state=seed)

			test_model(model=model,
					   trainer=trainer,
					   output_directory=train_directory,
					   output_prefix='train_sample',
					   dataset=train_dataset_sample,
					   data_df=train_df_sample,
					   output_html_sample=True,
					   output_combined_csv=script_config.get('output_combined_csv'),
					   overwrite_existing_output= script_config.get('overwrite_existing_output'),
					   )


	if 'additional_test' in dataset_config or 'additional' in dataset_config:
		additional_sets = []
		if 'additional_test' in dataset_config: #this is horribly ugly
			for d in dataset_config['additional_test']:
				d['path'] = d['test']
				d['suffix'] = 'test'
			additional_sets.extend(dataset_config['additional_test'])
		if 'additional' in dataset_config:
			additional_sets.extend(dataset_config['additional'])

		for additional_dataset_config in additional_sets:
			iprint(f'Doing additional test on {additional_dataset_config["name"]} ')


			additional_df, additional_dataset = read_dataset(
				additional_dataset_config['path'],
				model.tokenizer,
				additional_dataset_config['classes'],
				**read_dataset_config)

			test_model(model=model,
					   trainer=trainer,
					   output_directory=train_directory,
					   output_prefix=f'{additional_dataset_config["name"]}_{additional_dataset_config["suffix"]}',
					   dataset=additional_dataset,
					   data_df=additional_df,
					   output_html_sample=True,
					   output_combined_csv=True,
					   overwrite_existing_output= script_config.get('overwrite_existing_output'))



	iprint('Done training!')
	remove_log_destination(train_directory)

	return



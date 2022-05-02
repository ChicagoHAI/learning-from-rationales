import os
from traceback import print_exc

import pandas as pd
from pytorch_lightning import LightningModule
from lightning_component.trainer import ModifiedTrainer
from util.dataloader import RationaleDataloader
from util.dataset import RationaleDataset
from util.display_util import sample_and_output_as_html
from util.print_util import iprint
from typing import Sequence

seed = 83782

def test_model(model:LightningModule,
			   trainer:ModifiedTrainer,
			   output_prefix:str,
			   dataset:RationaleDataset,
			   output_directory:str,
			   data_df:pd.DataFrame,
			   output_combined_csv:bool=False,
			   output_html_sample:bool=False,
			   overwrite_existing_output:bool=False,
			   never_evaluate_rationales:bool=None):
	iprint(f'Doing full evaluation of model on set "{output_prefix}"')


	model.test_output_prefix = output_prefix

	existing_output_path = os.path.join(output_directory, f'{output_prefix}_output', f'epoch_-1_predictions.json')
	if not overwrite_existing_output:
		if os.path.exists(existing_output_path):
			try:
				existing_output_df = pd.read_json(existing_output_path, orient='records', lines=True)
				if existing_output_df.shape[0] == len(dataset):
					iprint('"overwrite_existing_output" set to False, and existing output matches length of dataset, so not running test function.')
					return
				else:
					iprint('"overwrite_existing_output" set to False, but existing output does not match length of dataset, so still running test function.')
			except Exception  as ex:
				print_exc()
				iprint('Error trying to load existing output. Continuing to run test function.')


	if never_evaluate_rationales is not None:
		prev_ner = model.never_evaluate_rationales
		model.never_evaluate_rationales = never_evaluate_rationales

	trainer.test(model=model, test_dataloaders=RationaleDataloader(
		dataset,
		pad_token_id=model.tokenizer.pad_token_id,
		batch_size=model.batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=True,
		sampler=None))

	# iprint(trainer.profiler.describe())
	if never_evaluate_rationales is not None:
		model.never_evaluate_rationales = prev_ner


	if output_html_sample or output_combined_csv:
		output_df = pd.read_json(existing_output_path, lines=True, orient='records')
		output_df['tokens'] = output_df['input_ids'].apply(model.tokenizer.convert_ids_to_tokens)


		# dev_output_df.to_json(os.path.join(train_directory, 'dev_predictions.json'), orient='records', lines=True)
		if output_html_sample:

			#select which model outputs get displayed as highlighted text, e.g. predicted rationale, human rationale

			rationalized_prediction_sets = aggregate_rationalized_prediction_column_sets(output_df.columns)

			sample_and_output_as_html(output_df,
									  os.path.join(output_directory, f'{output_prefix}_prediction_sample.html'),
									  rationalized_prediction_sets = rationalized_prediction_sets,
									  sample_function=lambda df: df.sample(n=min(100, output_df.shape[0]), random_state=seed),
									  color_predictions=True)

		if output_combined_csv:
			combined_output_df = pd.concat([data_df.iloc[0:output_df.shape[0]], output_df], axis=1)
			combined_output_df.to_csv(os.path.join(output_directory, f'{output_prefix}_output', f'combined_epoch_-1_predictions.csv'),index=False)

def aggregate_rationalized_prediction_column_sets(columns:Sequence[str]):
	'''
	Go through the set of columns outputted by the model and figure out combos of prediction/rationale

	Output list of dictionaries like the following:
	{
		'rationale':'predicted_rationale',
		'py_index':'py_index',
		'y_index':'label',
	}
	:param columns:
	:return:
	'''
	rationale_suffixes = ['_rationale',
						  '_importance',
						  '_rationale_losses',
						  # '_rationale_probs'
						  # '_rationale_diff'
						  '_rationale_loss_weight',
						  '_gradient',
						  '_gradients',
						  '_grad',
						  '_phi',
						  '_rationale_mask',
						  '_bert_attention'
						  ]
	rationalized_prediction_column_sets =[]
	for column in columns:
		for rationale_suffix in rationale_suffixes:
			if column.endswith(rationale_suffix):
				column_set = {'rationale':column, 'y_index':'label', 'tokens':'tokens'}
				column_prefix = column.split('_')[0]
				py_index_col = column_prefix+'_py_index'
				if py_index_col in columns:
					column_set['py_index'] = py_index_col
				elif column_prefix == 'predicted':
					column_set['py_index'] = 'py_index'
				else:
					column_set['py_index'] = None
				rationalized_prediction_column_sets.append(column_set)

	return rationalized_prediction_column_sets
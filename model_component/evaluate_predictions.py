from typing import Dict, Callable, List, Any
import pytorch_lightning as pl

from lightning_component.metrics import SampleMeanDeviation, fast_reset
from util.data_util import tensor_dict_to_numpy, invert_dict_list, combine_results_dict, model_output_to_df
import os
from util.print_util import iprint, now
from pprint import pformat
from util.misc_util import ensure_containing_dir, update_json_file
import numpy as np

'''
Functions pertaining to evaluating model output
'''


def retrieve_metric(model: pl.LightningModule,
					key: str,
					metric_cls: Callable,
					use_fast_reset=False,
					**metric_args):
	'''
	Create a pytorch lightning Metric object of the given class and attach it to the given model, or retrieve it if it already exists.
	:param model:
	:param key:
	:param metric_cls:
	:param metric_args:
	:return:
	'''
	retrieved_metrics = [(name, object) for name, object in model.named_modules() if name == key and isinstance(object, metric_cls)]
	if len(retrieved_metrics) == 1:
		return retrieved_metrics[0][1]
	elif len(retrieved_metrics) == 0:

		#Monkey-patch the faster reset function into the method class
		if use_fast_reset:
			metric_cls.reset = fast_reset
		metric = metric_cls(**metric_args).cuda(model.device)
		model.add_module(key, metric)
		model.auto_metrics[key] = metric


		return metric
	else:
		raise Exception(f'Found {len(retrieved_metrics)} metrics with name "{key}" and class {metric_cls}')


def evaluate_predictions(model,
						 batch: Dict,
						 result: Dict,
						 log_values: bool,
						 prefix: str = '',
						 compute_on_step=True,
						 result_prefix:str=''
						 ):
	'''
	Evaluate the prediction and rationale accuracy of a batch given a result, using the appropriate set of metrics (train or val)
	:param batch:
	:param result:
	:param mode:
	:return:
	'''

	evaluation_result = {}

	if 'label' in batch:
		accuracy = retrieve_metric(model, f'{prefix}accuracy', pl.metrics.Accuracy,compute_on_step=compute_on_step)
		evaluation_result[f'{prefix}accuracy'] = accuracy(result[result_prefix+'py_index'], batch['label'])

		f1 = retrieve_metric(model, f'{prefix}f1', pl.metrics.Fbeta,compute_on_step=compute_on_step)
		evaluation_result[f'{prefix}f1'] = f1(result[result_prefix+'py_index'], batch['label'])

		precision = retrieve_metric(model, f'{prefix}precision', pl.metrics.Precision,compute_on_step=compute_on_step)
		evaluation_result[f'{prefix}precision'] = precision(result[result_prefix+'py_index'], batch['label'])

		recall = retrieve_metric(model, f'{prefix}recall', pl.metrics.Recall,compute_on_step=compute_on_step)
		evaluation_result[f'{prefix}recall'] = recall(result[result_prefix+'py_index'], batch['label'])
		if log_values:
			# model.log(f'{prefix}loss', result[result_prefix+'loss'])
			model.log(f'{prefix}acc', evaluation_result[f'{prefix}accuracy'], prog_bar=True)
			# model.log(f'{prefix}f1', evaluation_result[f'{prefix}f1'], prog_bar=True)

	if 'human_rationale_sufficiency_accuracy' in result and 'predicted_human_rationale_sufficiency_accuracy' in result:
		HRSA_accuracy = retrieve_metric(model, f'{prefix}human_rationale_sufficiency_accuracy_prediction_accuracy', pl.metrics.Accuracy, compute_on_step=compute_on_step)
		evaluation_result[f'{prefix}human_rationale_sufficiency_accuracy_prediction_accuracy'] = HRSA_accuracy(result[result_prefix+'predicted_human_rationale_sufficiency_accuracy'], result[result_prefix+'human_rationale_sufficiency_accuracy'])

		HRSA_f1 = retrieve_metric(model, f'{prefix}human_rationale_sufficiency_accuracy_prediction_f1', pl.metrics.Fbeta, compute_on_step=compute_on_step)
		evaluation_result[f'{prefix}human_rationale_sufficiency_accuracy_prediction_f1'] = HRSA_f1(result[result_prefix+'predicted_human_rationale_sufficiency_accuracy'], result[result_prefix+'human_rationale_sufficiency_accuracy'])

	if 'sparsity_losses' in result:
		sparsity_deviation = retrieve_metric(model, f'{prefix}py_sparsity_deviation', SampleMeanDeviation, categories=list(range(model.num_classes)),compute_on_step=compute_on_step)
		evaluation_result[f'{prefix}py_sparsity_deviation'] = sparsity_deviation(result[result_prefix+'sparsity_losses'], result[result_prefix+'py_index'])



	# if log_values:
	# 	model.log(f'{prefix}loss', result[result_prefix+'loss'])
	# 	model.log(f'{prefix}acc', eval['accuracy'], prog_bar=True)
	# 	model.log(f'{prefix}f1', eval['f1'], prog_bar=True)
	#
	# 	if 'generator_loss' in result:
	# 		model.log('g_loss', result[result_prefix+'generator_loss'], prog_bar=True)
	# 	if 'predictor_loss' in result:
	# 		model.log('p_loss', result[result_prefix+'predictor_loss'], prog_bar=True)
	#
	# 	if 'sparsity_loss' in result:
	# 		model.log(f'{prefix}rat_sparsity', result[result_prefix+'sparsity_loss'], prog_bar=True)
	#
	# 	if 'py_sparsity_deviation' in eval:
	# 		model.log(f'{prefix}py_rat_deviance', eval['py_sparsity_deviation'], prog_bar=True)


	return evaluation_result



def evaluate_epoch(model, outputs: List[Any], set: str, epoch: int, verbose=True, no_file_io=False, write_evaluation:bool=True, write_predictions:bool=True):
	'''
	Accumulate the evaluations for a whole epoch, display them and optionally output them to a file
	:param model:
	:param outputs:
	:param set:
	:param epoch:
	:return:
	'''


	numpy_outputs = [tensor_dict_to_numpy(output) for output in outputs]
	output_dict = invert_dict_list(numpy_outputs)
	combined_output_dict = combine_results_dict(output_dict)
	epoch_evaluation = {'type': set, 'epoch': epoch, 'datetime': now(), 'step':model.global_step}

	# Collect the mean of every pointwise loss or fidelity value
	epoch_evaluation.update({key: np.nanmean(value) for key, value in combined_output_dict.items() if key.endswith('loss') or key.endswith('sufficiency') or key.endswith('comprehensiveness') or key.endswith('mean')})

	#Collect the aggregate value for every Metric we defined for this set over the course of evaluation.
	#Because of how the Metrics do accumuluation, this should provide the correct overall value for, e.g., rationale F1
	epoch_evaluation.update({key: metric.compute().detach().cpu().numpy().item() for key, metric in model.auto_metrics.items() if key.startswith(f'{set}/')})

	if verbose: iprint(pformat(epoch_evaluation))

	if model.output_dir is not None and not no_file_io:
		set_dir = os.path.join(model.output_dir, f'{set}_output')
		iprint(f'Writing {set} epoch {epoch} predictions and evaluation to {set_dir}')
		result_df = model_output_to_df(combined_output_dict, method='max')
		prediction_path = os.path.join(set_dir, f'epoch_{epoch}_predictions.json')
		ensure_containing_dir(prediction_path)
		# iprint(f'Outputting validation epoch {self.validation_epoch} results to {output_path}')
		if write_predictions:
			result_df.to_json(prediction_path, orient='records', lines=True)
		epoch_eval_filepath = os.path.join(set_dir, f'{set}_epoch_eval.json')
		if write_evaluation:
			update_json_file(epoch_evaluation, epoch_eval_filepath, epoch <= 0)

	return



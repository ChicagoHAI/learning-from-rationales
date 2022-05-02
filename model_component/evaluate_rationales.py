from typing import Dict

import pytorch_lightning as pl
import torch

from model_component.binarize import is_binary_rationale
from model_component.evaluate_predictions import retrieve_metric
from model_component.fidelity import generate_and_calculate_sufficiency_and_comprehensiveness
from model_component.masking import masked_flatten


def evaluate_rationale(prefix: str,
					   model: pl.LightningModule,
					   rationale_name: str,
					   rationale_tensor: torch.Tensor,
					   batch: Dict,
					   baseline_result: Dict,
					   evaluate_metrics: bool = True,
					   evaluate_fidelity: bool = False,
					   num_quantiles: int = 10):
	evaluation_result = {}
	if is_binary_rationale(rationale_name, model):
		evaluation_result.update(evaluate_rationale_at_threshold(prefix=prefix,
																 model=model,
																 rationale_name=rationale_name,
																 rationale_tensor=rationale_tensor,
																 batch=batch,
																 baseline_result=baseline_result,
																 evaluate_metrics=evaluate_metrics,
																 evaluate_fidelity=evaluate_fidelity,
																 threshold=0.5))
	else:
		#if it is a continous explanation like simple saliency, threshold it at a bunch of different quantiles (e.g. 10%, 20%), and evaluate each quantile separately. Then later
		#on we can draw a curve and look at the area under it.
		quantiles = torch.arange(0, 1 + 1 / num_quantiles, 1 / num_quantiles, device=rationale_tensor.device)
		#Don't take masked values into account when defining the quantile ranges
		p_alpha_quantiles = torch.quantile(masked_flatten(rationale_tensor.float(), mask = baseline_result['padding_mask']), quantiles)
		p_alpha_quantiles[-1] += .001 #So that the 10th decile catches everything, not everything but the max element
		for i in range(len(quantiles)):
			evaluation_result.update(evaluate_rationale_at_threshold(prefix=prefix,
																	 model=model,
																	 rationale_name=rationale_name,
																	 rationale_tensor=rationale_tensor,
																	 batch=batch,
																	 baseline_result=baseline_result,
																	 suffix=f'_@{(100 * (1-quantiles[i])).round().int()}%',
																	 evaluate_metrics=evaluate_metrics,
																	 evaluate_fidelity=evaluate_fidelity,
																	 threshold=p_alpha_quantiles[i],
																	 threshold_func= lambda x, t: (x >= t).float()))
			pass


	return evaluation_result


def evaluate_rationale_at_threshold(model: pl.LightningModule,
									rationale_name: str,
									rationale_tensor:torch.Tensor,
									batch: Dict,
									baseline_result: Dict,
									prefix: str = '',
									suffix: str = '',
									evaluate_metrics: bool = True,
									evaluate_fidelity: bool = False,
									threshold=0.5,
									threshold_func = lambda x, t: (x >= t).float()):
	'''
	yes:
	accuracy, recall, precision, f1
	fidelity

	no:
	sparsity
	other losses

	:param p_alpha:
	:param rationale:
	:return:
	'''

	evaluation_result = {}
	if evaluate_metrics:
		flattened_true_rationale = masked_flatten(batch['human_rationale'], batch['human_rationale_weight'])
		flattened_p_alpha = masked_flatten(rationale_tensor, batch['human_rationale_weight'])
		flattened_p_alpha = threshold_func(flattened_p_alpha, threshold)

		accuracy = retrieve_metric(model, f'{prefix}{rationale_name}{suffix}/accuracy', pl.metrics.Accuracy)
		evaluation_result[f'{prefix}{rationale_name}{suffix}/accuracy'] = accuracy(flattened_p_alpha, flattened_true_rationale)

		f1 = retrieve_metric(model, f'{prefix}{rationale_name}{suffix}/f1', pl.metrics.Fbeta)
		evaluation_result[f'{prefix}{rationale_name}{suffix}/f1'] = f1(flattened_p_alpha, flattened_true_rationale)

		precision = retrieve_metric(model, f'{prefix}{rationale_name}{suffix}/precision', pl.metrics.Precision)
		evaluation_result[f'{prefix}{rationale_name}{suffix}/precision'] = precision(flattened_p_alpha, flattened_true_rationale)

		recall = retrieve_metric(model, f'{prefix}{rationale_name}{suffix}/recall', pl.metrics.Recall)
		evaluation_result[f'{prefix}{rationale_name}{suffix}/recall'] = recall(flattened_p_alpha, flattened_true_rationale)
		evaluation_result[f'{prefix}{rationale_name}{suffix}/mean']= flattened_p_alpha.mean()

	if evaluate_fidelity:
		fidelity_result = generate_and_calculate_sufficiency_and_comprehensiveness(model=model,
																				   rationale_name=rationale_name,
																				   rationale_tensor=rationale_tensor,
																				   batch=batch,
																				   baseline_result=baseline_result,
																				   threshold=threshold,
																				   threshold_func=threshold_func,
																				   prefix=prefix,
																				   suffix=suffix)
		evaluation_result.update(fidelity_result)

	# evaluation_result = {key: value.detach().cpu() for key, value in evaluation_result.items()}
	return evaluation_result
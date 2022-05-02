import torch
from typing import Dict
# from model_component.evaluate import is_binary_rationale
from model_component.binarize import is_binary_rationale
# from model_component.evaluate import evaluate_predictions
from model_component.evaluate_predictions import evaluate_predictions


'''
Functions for calculating sufficiency and comprehensiveness (TKTK) of rationales
'''


def calculate_output_differences(primary_outputs: torch.Tensor, reduced_outputs: torch.Tensor):
	'''
	Calculate the difference in two sets of outputs. Will usually be class probabilities, but could also be class logits.

	Specifically, look at the difference for the max primary output. E.g. if these are class probabilities,
	look at the difference in the probability for the predicted (max prob) class for each row.
	:param primary_outputs:
	:param reduced_outputs:
	:return:
	'''

	max_output_indices = torch.argmax(primary_outputs, dim=1).unsqueeze(1)
	max_outputs = torch.gather(primary_outputs, 1, max_output_indices)
	max_reduced_outputs = torch.gather(reduced_outputs, 1, max_output_indices)
	return torch.abs(max_outputs - max_reduced_outputs).squeeze(1)


def test_calculate_output_differences():
	print('Testing calculate_output_differences.')
	primary_py_probs = torch.tensor([[0, 1], [1, 0], [0.5, 0.5], [0.25, 0.75]])
	reduced_py_probs = torch.tensor([[1, 0], [0.25, 0.75], [0, 1], [0.5, 0.5]])
	output_differences = torch.tensor([1.0, 0.75, 0.5, 0.25])

	assert torch.all(calculate_output_differences(primary_py_probs, reduced_py_probs) == output_differences)
	print('Passed!')


class FakeModel(torch.nn.Module):
	'''
	Fake "model" that classifies each input row as even or odd, with probabilities
	equal to even/odd ratios and "logits" equal to the raw counts of even/odd values.
	'''

	def forward(self,
				input_ids: torch.Tensor,
				padding_mask: torch.Tensor = None,
				label: torch.Tensor = None,
				rationale: torch.Tensor = None,
				rationale_weight: torch.Tensor = None,
				p_alpha: torch.Tensor = None):

		if p_alpha is not None:
			combined_mask = padding_mask * p_alpha
		else:
			combined_mask = padding_mask

		modded_ids = torch.fmod(input_ids, 2)
		even_counts = (modded_ids * combined_mask).sum(dim=1)
		odd_counts = ((1 - modded_ids) * combined_mask).sum(dim=1)

		result = {}
		result['py_logits'] = torch.stack([even_counts, odd_counts], dim=1)
		result['py_probs'] = result['py_logits'] / combined_mask.sum(dim=1).unsqueeze(1)
		result['py_index'] = torch.argmax(result['py_probs'], dim=1)

		return result


def generate_and_calculate_fidelities(model: torch.nn.Module,
									  p_alpha_key: str,
									  batch: Dict,
									  result: Dict,
									  num_quantiles=10):
	fidelity_result = {}
	if is_binary_rationale(p_alpha_key, model):
		fidelity_result.update(generate_and_calculate_sufficiency_and_comprehensiveness(model=model,
																						rationale_name=p_alpha_key,
																						batch=batch,
																						baseline_result=result,
																						threshold=0.5))
	else:
		quantiles = torch.arange(0, 1, 1 / num_quantiles, device=result[p_alpha_key].device)
		p_alpha_quantiles = torch.quantile(result[p_alpha_key], quantiles)
		for i in range(num_quantiles):
			fidelity_result.update(generate_and_calculate_sufficiency_and_comprehensiveness(model=model,
																							rationale_name=p_alpha_key,
																							batch=batch,
																							baseline_result=result,
																							suffix=f'_@{int(100 * quantiles[i])}%',
																							threshold=p_alpha_quantiles[i]))

	return fidelity_result


#
# from model.bert_softmax_attention_model import BertSoftmaxAttentionModel
# from model.base_rationale_model import BaseRationaleModel


def generate_and_calculate_sufficiency_and_comprehensiveness(model: torch.nn.Module,
															 rationale_name: str,
															 rationale_tensor: torch.Tensor,
															 batch: Dict,
															 baseline_result: Dict,
															 prefix: str = '',
															 suffix: str = '',
															 threshold: float = 0.5,
															 threshold_func=lambda x, t: (x >= t).float()):
	sufficiency_dict = generate_and_calculate_fidelity(model=model,
													   batch=batch,
													   baseline_result=baseline_result,
													   rationale_name=rationale_name,
													   rationale_tensor=rationale_tensor,
													   fidelity_name='sufficiency',
													   invert_difference=True,
													   invert_p_alpha=False,
													   threshold=threshold,
													   threshold_func=threshold_func,
													   return_probs=True,
													   prefix=prefix,
													   suffix=suffix)

	return sufficiency_dict

	#todo fix this
	# for now, not really interested in comprehensiveness.
	# comprehensiveness, comprehensiveness_probs = generate_and_calculate_fidelity(model=model,
	# 																			 batch=batch,
	# 																			 baseline_result=baseline_result,
	# 																			 rationale_name=rationale_name,
	# 																			 rationale_tensor=rationale_tensor,
	# 																			 fidelity_name = 'comprehensiveness',
	# 																			 invert_difference=False,
	# 																			 invert_p_alpha=True,
	# 																			 threshold=threshold,
	# 																			 threshold_func=threshold_func,
	# 																			 return_probs=True,
	# 																			 prefix=prefix,
	# 																			 suffix=suffix)
	#
	# return {f'{prefix}{rationale_name}{suffix}/sufficiency': sufficiency,
	# 		f'{prefix}{rationale_name}{suffix}/sufficiency_probs': sufficiency_probs,
	# 		f'{prefix}{rationale_name}{suffix}/comprehensiveness': comprehensiveness,
	# 		f'{prefix}{rationale_name}{suffix}/comprehensiveness_probs': comprehensiveness_probs}


def generate_and_calculate_fidelity(model: torch.nn.Module,
									batch: Dict,
									baseline_result: Dict,
									rationale_name: str,
									rationale_tensor: torch.Tensor,
									fidelity_name:str,
									invert_difference: bool = False,
									invert_p_alpha: bool = False,
									threshold: float = 0.5,
									threshold_func=lambda x, t: (x >= t).float(),
									return_probs: bool = False,
									calculate_accuracy: bool = True,
									prefix:str='',
									suffix:str=''):
	'''
	Take a predicted p_alpha and set of py_probs, and use them to generate a new (masked) prediction.

	Then use the new prediction to calculate the fidelity of p_alpha with respect to the original prediction.

	Then maybe take a nap.

	:param model:
	:param py_probs:
	:param p_alpha:
	:param input_ids:
	:param padding_mask:
	:param invert_p_alpha: whether to invert p_alpha before passing it in. Make this true for comprehensiveness and false for sufficiency
	:return:
	'''

	# p_alpha = rationale_tensor


	rationale_tensor = threshold_func(rationale_tensor, threshold)

	if invert_p_alpha:
		rationale_tensor = 1 - rationale_tensor

	new_result = model(input_mask=rationale_tensor,
					   **batch)

	if calculate_accuracy:
		#this will log these evaluation metrics to the model, where they will get aggregated and calculated in BaseModel.test_epoch_end()
		prediction_eval = evaluate_predictions(model=model,
							 batch=batch,
							 result=new_result,
							 log_values=False,
							 prefix=f'{prefix}{rationale_name}{suffix}/{fidelity_name}_',
						   compute_on_step=False)

	output_differences = calculate_output_differences(baseline_result['py_probs'], new_result['py_probs'])

	if invert_difference:
		output_differences = 1 - output_differences

	# if return_probs:
	# 	return output_differences, new_result['py_probs']
	#
	# else:
	# 	return output_differences

	#return all losses, as well as the calculated fidelity
	rdict = {f'{prefix}{rationale_name}{suffix}/{fidelity_name}_{key}':value for key, value in new_result.items() if key.endswith('loss')}
	rdict[f'{prefix}{rationale_name}{suffix}/{fidelity_name}_py_probs'] = new_result['py_probs']
	rdict[f'{prefix}{rationale_name}{suffix}/{fidelity_name}_py_index'] = new_result['py_index']


	rdict[f'{prefix}{rationale_name}{suffix}/{fidelity_name}'] = output_differences
	return rdict


def test_generate_and_calculate_fidelity():
	print('Testing fidelity generation/calculation')
	model = FakeModel()
	input_ids = torch.tensor([[1, 2, 3, 4],
							  [1, 3, 5, 7],
							  [2, 4, 6, 8],
							  [1, 3, 5, 6]])

	padding_mask = torch.tensor([[1, 1, 1, 1],
								 [1, 1, 1, 0],
								 [1, 1, 0, 0],
								 [1, 1, 1, 1]])

	py_probs = torch.tensor([[0.5, 0.5],
							 [0.0, 1.0],
							 [1.0, 0.0],
							 [0.25, 0.75]])

	result = model(input_ids=input_ids,
				   padding_mask=padding_mask)

	p_alpha = torch.tensor([[1, .97, .78, 0],
							[1, 1, 0, 0],
							[1, 0, 0, 0],
							[1, 0, 0, 1]])

	reduced_py_probs = torch.tensor([[1 / 3, 2 / 3],
									 [0.0, 1.0],
									 [1.0, 0.0],
									 [0.5, 0.5]])

	inverted_p_alpha = torch.tensor([[0, 0.03, 0.22, 1],
									 [0, 0, 1, 1],
									 [0, 1, 1, 1],
									 [0, 1, 1, 0]])

	inverted_reduced_py_probs = torch.tensor([[1.0, 0.0],
											  [0.0, 1.0],
											  [1.0, 0.0],
											  [0.0, 1.0]])

	# reduced_result = model(input_ids=input_ids,
	# 					   padding_mask=padding_mask,
	# 					   p_alpha=p_alpha)

	output_differences = generate_and_calculate_fidelity(model=model,
														 py_probs=result['py_probs'],
														 p_alpha=p_alpha,
														 input_ids=input_ids,
														 padding_mask=padding_mask,
														 invert_p_alpha=False,
														 reduction=None)
	target_output_differences = torch.tensor([1 / 6, 0.0, 0.0, 0.25])
	assert torch.all(torch.abs(output_differences - target_output_differences) < 0.0001)

	inverted_output_differences = generate_and_calculate_fidelity(model=model,
																  py_probs=result['py_probs'],
																  p_alpha=p_alpha,
																  input_ids=input_ids,
																  padding_mask=padding_mask,
																  invert_p_alpha=True,
																  reduction=None)
	target_inverted_output_differences = torch.tensor([0.5, 0.0, 0.0, 0.25])
	assert torch.all(torch.abs(inverted_output_differences - target_inverted_output_differences) < 0.0001)

	print('Passed!')


def main():
	test_calculate_output_differences()
	test_generate_and_calculate_fidelity()


if __name__ == '__main__':
	main()

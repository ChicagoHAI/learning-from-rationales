'''
Utility functions pertaining to producing binary rationales
'''
import torch

from model_component.gumbel_softmax import gumbel_softmax
from util.print_util import iprint


def is_binary_rationale(rationale_key: str, strict=False):
	'''
	Hacky way to determine whether a rationale is binary or continous. Cleaner would be to base it on the model class, but that leads to circular import issues
	:param rationale_key:
	:param model:
	:return:
	'''

	rationale_key = rationale_key.replace('_into_generator','')

	# if rationale_key in ['human_rationale', 'predicted_rationale', 'prediction_rationale','rationale_full_info', 'rationale_no_info']:
	if rationale_key.endswith('rationale'):
		return True
	elif rationale_key.endswith('importance'):
		return True
	elif rationale_key.endswith('mask'):
		return True
	elif rationale_key.endswith('gradient'):
		return False
	elif rationale_key.endswith('gradients'):
		return False
	elif rationale_key.endswith('grad'):
		return False
	elif rationale_key.endswith('_c_'):
		return False
	elif rationale_key.endswith('attribution'):
		return False
	elif rationale_key.endswith('phi'):
		return False
	elif rationale_key.endswith('losses'):
		return False
	elif rationale_key.endswith('attention'):
		return False


	iprint(f'Cannot determine whether rationale {rationale_key} is binary or continuous. Returning False by default')
	return False


def phi_to_rationale(phi: torch.Tensor,
					 binarization_method:str,
					 training:bool=True,
					 gumbel_train_temperature:float=None,
					 gumbel_eval_temperature:float=None,
					 ):
	if binarization_method == 'gumbel_softmax':
		# zero_phi = torch.log(1 - self.sigmoid(phi))

		if training:
			zero_phi = torch.zeros_like(phi)
			# mathematically this works out to a situation where the probabilities (which the gumbel softmax will approximate sampling from) will be = sigmoid(phi)
			both_phi = torch.stack([zero_phi, phi], dim=2)
			temperature = gumbel_train_temperature if training else gumbel_eval_temperature
			predicted_rationale = gumbel_softmax(both_phi, tau=temperature, dim=2)[:, :, 1]
			predicted_rationale = torch.nan_to_num(predicted_rationale, nan=0.0, posinf=0.0, neginf=0.0)
		else:
			predicted_rationale = torch.sigmoid(phi).round()

	# phi = phi[:, :, 1]
	elif binarization_method == 'bernoulli':
		predicted_rationale_probs = torch.sigmoid(phi)
		# result['predicted_rationale_c_probs'] = predicted_rationale_probs
		predicted_rationale = torch.bernoulli(predicted_rationale_probs).detach()
	elif binarization_method == 'sigmoid':
		predicted_rationale = torch.sigmoid(phi)

	return predicted_rationale
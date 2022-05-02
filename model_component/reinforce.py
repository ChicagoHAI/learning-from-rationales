import torch
from model_component.masking import masked_sum, masked_mean


def calculate_reinforce_losses(sampled_z: torch.Tensor,
							   z_logits: torch.Tensor,
				   sampled_losses: torch.Tensor,
				   mask: torch.Tensor,
				   return_log_probs=False):
	'''
	Given sampled binary z, total loss based on that z, and probabilities of z = 1, calculate a REINFORCE loss term
	that we can get the gradient for
	:param sampled_z: [batch size x batch width], dtype = float
	:param z_probs: [batch size x batch width], dtype = float; bernoulli probabilities from which z was sampled
	:param loss: [batch_size], dtype=float
	:param padding_mask: [batch size x batch width], dtype = long; 0 where we should ignore sampled z values for the purpose of the estimated gradient
	:return:
	reinforce_loss: [batch size], dtype=float
	'''
	assert len(sampled_losses.shape) == 1, 'Loss must be batch size x 0'
	assert len(z_logits.shape) == 2, 'Probs must be batch size x batch width'

	sampled_log_probs = -torch.nn.functional.binary_cross_entropy_with_logits(z_logits, sampled_z, reduction='none')
	sampled_log_prob_sums = masked_sum(sampled_log_probs, mask=mask, dim=1)

	reinforce_losses = sampled_losses.detach() * sampled_log_prob_sums

	if return_log_probs:
		return reinforce_losses, sampled_log_prob_sums
	else:
		return reinforce_losses


# sampled_log_probs = -torch.nn.functional.binary_cross_entropy_with_logits(result['phi'].squeeze(2), result['p_alpha'], reduction='none')

def test_reinforce_loss():
	sampled_z = torch.Tensor([[1, 0, 1, 0],
							  [0, 0, 1, 1],
							  [1, 1, 1, 0],
							  [1, 0, 1, 1]])
	z_probs = torch.Tensor([[.6, .2, .8, .1],
							[.3, .7, .7, .8],
							[.5, .9, .3, .2],
							[.6, .4, .9, .4]])
	z_logits = torch.logit(z_probs)
	sampled_z_probs = torch.Tensor([[.6, .8, .8, .9],
									[.7, .3, .7, .8],
									[.5, .9, .3, .8],
									[.6, .6, .9, .4]])
	mask = torch.Tensor([[1, 1, 1, 0],
						 [1, 1, 1, 1],
						 [1, 1, 0, 0],
						 [1, 1, 1, 0]])
	log_prob_sums = masked_sum(torch.log(sampled_z_probs), mask, dim=1)
	sparsity_losses = masked_mean(sampled_z,mask=mask, dim=1)
	losses = sparsity_losses
	reinforce_losses = losses.detach() * log_prob_sums

	f_reinforce_losses, f_log_prob_sums = calculate_reinforce_losses(sampled_z=sampled_z, z_logits = z_logits, sampled_losses=losses, mask=mask, return_log_probs=True)

	print('Testing reinforce loss calculation')
	assert torch.all(f_reinforce_losses==reinforce_losses)
	assert torch.all(f_log_prob_sums == log_prob_sums)
	print(f_reinforce_losses)
	print(f_log_prob_sums)
	print('Done')



if __name__ == '__main__':
	test_reinforce_loss()

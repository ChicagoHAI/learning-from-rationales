import torch
from typing import Optional, Any, List
from pytorch_lightning.metrics import Metric, Accuracy

'''
Various Metric classes defined to fit the Pytorch Lightning API

The Metric API is a little weird, but the way it should be used is:

metric.forward(epoch 0 batch 0) #calculates metric for this batch, and accumulates values as necessary (e.g. #correct vs #total for accuracy)
metric.forward(epoch 0 batch 2) 
...
metric.forward(epoch 0 batch N) 
metric.compute() #Calculates cumulative metric for all N batches

metric.forward(epoch 1 batch 0) # Resets accumulated values, starting fresh from defaults
...



'''


class Mean(Metric):
	'''
	Very simple metric that just calculates a mean
	:param Metric:
	:return:
	'''

	def __init__(
			self,
			compute_on_step: bool = True,
			dist_sync_on_step: bool = False,
			process_group: Optional[Any] = None,
	):
		super().__init__(
			compute_on_step=compute_on_step,
			dist_sync_on_step=dist_sync_on_step,
			process_group=process_group,
		)

		self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
		self.add_state("size", default=torch.tensor(0), dist_reduce_fx="sum")

	def update(self, t: torch.Tensor):
		"""
		Update state with sum and size

		Args:
			t: float tensor
		"""

		self.sum += torch.sum(t)
		self.size += t.numel()

	def compute(self):
		"""
		Computes mean over state.
		"""
		return self.sum.float() / self.size


def test_Mean_metric():
	print('Testing Mean metric')
	t1 = torch.tensor([0,1,1,0])
	t2 = torch.tensor([1,1,1,1])
	t3 = torch.tensor([0,0,0,1])
	t4 = torch.tensor([0,0,0,0])

	meanm = Mean()

	assert torch.isnan(meanm.compute())

	assert meanm(t1) == 0.5
	assert meanm(t2) == 1.0
	assert meanm.compute() == 0.75

	assert meanm(t3) == 0.25
	assert meanm.compute() == 0.25

	assert meanm(t4) == 0
	assert meanm(t2) == 1.0
	assert meanm.compute() == 0.5


	print('Success!')

class SampleMeanDeviation(Metric):
	'''
	Very simple metric of association between categorical and continuous variable
	'''
	def __init__(
			self,
			categories:List,
			compute_on_step: bool = True,
			dist_sync_on_step: bool = False,
			process_group: Optional[Any] = None,
	):
		super().__init__(
			compute_on_step=compute_on_step,
			dist_sync_on_step=dist_sync_on_step,
			process_group=process_group,
		)

		self.categories = categories

		for category in categories:
			self.add_state(f"group_{category}", default=[], dist_reduce_fx=None)

	def update(self, cont_values: torch.Tensor, cat_values:torch.tensor):
		"""
		Update state with sum and size

		Args:
			t: float tensor
		"""

		for category in self.categories:
			getattr(self,f"group_{category}").append(cont_values[torch.nonzero(cat_values == category)])

		return


	def compute(self):
		"""
		Computes max deviation between sample means, as a fraction of the overall population mean
		"""
		all_means = []
		# counts = []
		for category in self.categories:
			combined = torch.cat(getattr(self,f"group_{category}"))
			all_means.append(combined.mean())
			# counts.append(combined.numel())

		# overall_mean = sum([count*mean for mean, count in zip(means,counts) if count > 0])/sum(counts)

		means = [mean for mean in all_means if not torch.isnan(mean)]

		deviances = []
		for i in range(len(means)):
			for j in range(i+1, len(means)):
				deviances.append(torch.abs(means[i]-means[j])/(means[i]+means[j]))

		if len(deviances) > 0:
			value = torch.max(torch.stack(deviances))
		else:
			value = torch.tensor(0.0, device=all_means[0].device)


		return value

from copy import deepcopy
def fast_reset(self):
	"""
	The base Metric.reset() function does a tensor.to() operation every time it is called, which is very slow.
	This version just moves the default value to the correct device once, on the basis that the device isn't going to change over the course of training or evaluation or whatever
	"""

	for attr in self._defaults.keys():
		current_val = getattr(self, attr)

		if isinstance(current_val, torch.Tensor):
			if not self._defaults[attr].device == current_val.device:
				self._defaults[attr] = self._defaults[attr].to(current_val.device)
			# setattr(self, attr, self._defaults[attr].clone())
			current_val.copy_(self._defaults[attr])
		else:
			setattr(self, attr, deepcopy(self._defaults[attr]))

	return

def main():
	test_Mean_metric()


if __name__ == '__main__':
	main()
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn


class ModifiedTensorboardLogger(pl_loggers.TensorBoardLogger):
	'''
	Version of the pytorch tensorboard logger that handles graph logging differently
	'''

	@rank_zero_only
	def log_graph(self, model: LightningModule, **kwargs):
		'''
		Log the graph using the training step as the forward function and the first training batch as the input array.
		:param model:
		:param kwargs:
		:return:
		'''
		if self._log_graph:

			first_training_batch = model.transfer_batch_to_device(next(iter(model.train_dataloader())), model.device)
			# input_array = model.transfer_batch_to_device(input_array, model.device)
			model_forward = model.forward
			# model.forward = lambda b:model.training_step(b,  batch_idx=0, forward_function = model_forward, only_forward=True)
			model.forward = lambda b:model_forward(**b)['py_logits']

			self.experiment.add_graph(model, first_training_batch)
			model.forward = model_forward

		return
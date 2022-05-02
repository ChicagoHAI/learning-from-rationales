from typing import Optional, Union

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.training_loop import TrainLoop

from lightning_component.tuner import SafeTuner


class ModifiedTrainer(Trainer):
	'''
	Subclass of Pytorch Lightning trainer with slight tweaks
	'''

	def __init__(self, *args,
				 auto_scale_safety_factor:float=0.1,
				 max_epochs: int = 1000,
				 min_epochs: int = 1,
				 max_steps: Optional[int] = None,
				 min_steps: Optional[int] = None,
				 num_sanity_val_steps: int = 2,
				 automatic_optimization: bool = True,
				 auto_lr_find: Union[bool, str] = False,
				 auto_scale_batch_size: Union[str, bool] = False,
				 **kwargs):
		super().__init__(*args, **kwargs)
		self.tuner = SafeTuner(self, safety_factor=auto_scale_safety_factor)
		self.train_loop = ModifiedTrainLoop(self)

		# configure train loop
		self.train_loop.on_trainer_init(
			max_epochs,
			min_epochs,
			max_steps,
			min_steps,
			num_sanity_val_steps,
			automatic_optimization
		)

		# configure tuner
		self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)


class ModifiedTrainLoop(TrainLoop):

	def reset_train_val_dataloaders(self, model):
		if not self.trainer.reload_dataloaders_every_epoch:
			self.trainer.reset_train_dataloader(model)

		#I think this is a bug in a recent version of pytorch lightning. This function should really reset both dataloaders regardless.
		# if self.trainer.val_dataloaders is None and not self.trainer.reload_dataloaders_every_epoch:
		if not self.trainer.reload_dataloaders_every_epoch:
			self.trainer.reset_val_dataloader(model)
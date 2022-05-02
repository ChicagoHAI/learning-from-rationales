import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.early_stopping import TPU_AVAILABLE


class VerboseEarlyStopping(EarlyStopping):
	def _run_early_stopping_check(self, trainer, pl_module):
		"""
		Checks whether the early stopping condition is met
		and if so tells the trainer to stop the training.
		"""
		logs = trainer.logger_connector.callback_metrics

		if not self._validate_condition_metric(logs):
			return  # short circuit if metric not present

		current = logs.get(self.monitor)

		# when in dev debugging
		trainer.dev_debugger.track_early_stopping_history(self, current)

		if not isinstance(current, torch.Tensor):
			current = torch.tensor(current, device=pl_module.device)

		if trainer.use_tpu and TPU_AVAILABLE:
			current = current.cpu()

		if self.monitor_op(current - self.min_delta, self.best_score):
			log.info(f'Better score {current:.3f} than current best {self.best_score:.3f} found at epoch {trainer.current_epoch} / step {trainer.global_step}; resetting wait count to 0')
			self.best_score = current
			self.wait_count = 0
		else:
			self.wait_count += 1
			should_stop = self.wait_count >= self.patience
			log.info(f'Not-better score {current:.3f} than current best {self.best_score:.3f} found at epoch {trainer.current_epoch} / step {trainer.global_step}; wait count set to {self.wait_count}')

			if bool(should_stop):
				self.stopped_epoch = trainer.current_epoch
				log.info(f'Early-stopping trainer at epoch {self.stopped_epoch} / step {trainer.global_step} due to no improvement in {self.patience} validation epochs over current best {self.best_score:.3f}')
				trainer.should_stop = True

		# stop every ddp process if any world process decides to stop
		should_stop = trainer.accelerator_backend.early_stopping_should_stop(pl_module)
		trainer.should_stop = should_stop
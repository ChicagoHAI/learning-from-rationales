import os

from pytorch_lightning import _logger as log, LightningModule
from pytorch_lightning.tuner.batch_size_scaling import _adjust_batch_size, _run_power_scaling
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_hasattr
from pytorch_lightning.loggers.base import DummyLogger

from lightning_component.stopper import VerboseEarlyStopping

class SafeTuner(Tuner):
	def __init__(self, *args, safety_factor:float=0.1,  **kwargs):
		super().__init__(*args, **kwargs)
		self.safety_factor = safety_factor



	# def scale_batch_size(self,
	# 					 model,
	# 					 **kwargs):
	def scale_batch_size(self,
						 model,
						 mode: str = 'power',
						 steps_per_trial: int = 3,
						 init_val: int = 2,
						 max_trials: int = 6,
						 batch_arg_name: str = 'batch_size',
						 **fit_kwargs):
		# new_size = super().scale_batch_size(model, **kwargs)
		# return safer_scale_batch_size(
		# 	self.trainer, model, mode, steps_per_trial, init_val, max_trials, batch_arg_name, **fit_kwargs
		# )

		new_size= safer_scale_batch_size(
			self.trainer, model, mode, steps_per_trial, init_val, max_trials, batch_arg_name,**fit_kwargs
		)


		new_new_size = new_size-int(new_size*self.safety_factor)
		log.info(f'Reducing calculated batch size {new_size} by safety factor of {self.safety_factor} to {new_new_size}')
		model.batch_size = new_new_size
		return new_new_size


def safer_scale_batch_size(trainer,
					 model: LightningModule,
					 mode: str = 'power',
					 steps_per_trial: int = 1,
					 init_val: int = 2,
					 max_trials: int = 25,
					 batch_arg_name: str = 'batch_size',
					 **fit_kwargs):
	r"""
	Version of this function that runs a safer version of the binsearch
	"""
	if not lightning_hasattr(model, batch_arg_name):
		raise MisconfigurationException(
			f'Field {batch_arg_name} not found in both `model` and `model.hparams`')
	if hasattr(model, batch_arg_name) and hasattr(model, "hparams") and batch_arg_name in model.hparams:
		rank_zero_warn(
			f'Field `model.{batch_arg_name}` and `model.hparams.{batch_arg_name}` are mutually exclusive!'
			f' `model.{batch_arg_name}` will be used as the initial batch size for scaling.'
			f' If this is not the intended behavior, please remove either one.'
		)

	if hasattr(model.train_dataloader, 'patch_loader_code'):
		raise MisconfigurationException('The batch scaling feature cannot be used with dataloaders'
										' passed directly to `.fit()`. Please disable the feature or'
										' incorporate the dataloader into the model.')

	# Arguments we adjust during the batch size finder, save for restoring
	__safe_scale_batch_dump_params(trainer)

	# Set to values that are required by the algorithm
	__safe_scale_batch_reset_params(trainer, model, steps_per_trial)

	# Save initial model, that is loaded after batch size is found
	save_path = os.path.join(trainer.default_root_dir, 'temp_model.ckpt')
	trainer.save_checkpoint(str(save_path))

	if trainer.progress_bar_callback:
		trainer.progress_bar_callback.disable()

	# Initially we just double in size until an OOM is encountered
	new_size = _adjust_batch_size(trainer, value=init_val)  # initially set to init_val
	if mode == 'power':
		new_size = _run_power_scaling(trainer, model, new_size, batch_arg_name, max_trials, **fit_kwargs)
	elif mode == 'binsearch':
		new_size = _run_safer_binsearch_scaling(trainer, model, new_size, batch_arg_name, max_trials,  **fit_kwargs)
	else:
		raise ValueError('mode in method `scale_batch_size` can only be `power` or `binsearch')

	garbage_collection_cuda()
	log.info(f'Finished batch size finder, will continue with full run using batch size {new_size}')

	# Restore initial state of model
	trainer.checkpoint_connector.restore(str(save_path), on_gpu=trainer.on_gpu)
	os.remove(save_path)

	# Finish by resetting variables so trainer is ready to fit model
	__safe_scale_batch_restore_params(trainer)
	if trainer.progress_bar_callback:
		trainer.progress_bar_callback.enable()

	return new_size


def _run_safer_binsearch_scaling(trainer, model, new_size, batch_arg_name, max_trials, **fit_kwargs):
	"""
	Version of this function that does a final fit using the final concluded batch size.

	Reason: I've found that if the last batch in this process is an OOM error, this causes memory leaks from unaddressed gradients left over in the computational graph
	"""
	high = None
	count = 0
	while True:
		garbage_collection_cuda()
		trainer.global_step = 0  # reset after each try
		try:
			# Try fit
			trainer.fit(model, **fit_kwargs)
			count += 1
			if count > max_trials:
				break
			# Double in size
			low = new_size
			if high:
				if high - low <= 1:
					break
				midval = (high + low) // 2
				new_size, changed = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc='succeeded')
			else:
				new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc='succeeded')

			if not changed:
				break

		except RuntimeError as exception:
			# Only these errors should trigger an adjustment
			if is_oom_error(exception):
				# If we fail in power mode, half the size and return
				garbage_collection_cuda()
				high = new_size
				midval = (high + low) // 2
				new_size, _ = _adjust_batch_size(trainer, value=midval, desc='failed')
				if high - low <= 1:
					# trainer.fit(model, **fit_kwargs) #do a last fit with the final batch size
					break
					# pass
				# else:
				# 	print()
			else:
				raise  # some other error not memory related


	garbage_collection_cuda()
	# model.zero_grad()
	# # try:
	# model.optimizers().step()
	# model.optimizers().zero_grad()
	# except:
	# 	pass

	# model.batch_size = model.batch_size // 2
	# print(f'Confirming batch size {model.batch_size}')
	# trainer.global_step = 0
	# trainer.fit(model, **fit_kwargs)

	return new_size


def __safe_scale_batch_dump_params(trainer):
	# Prevent going into infinite loop
	trainer.__dumped_params = {
		'auto_lr_find': trainer.auto_lr_find,
		'current_epoch': trainer.current_epoch,
		'max_steps': trainer.max_steps,
		'weights_summary': trainer.weights_summary,
		'logger': trainer.logger,
		'callbacks': trainer.callbacks,
		'checkpoint_callback': trainer.checkpoint_callback,
		'auto_scale_batch_size': trainer.auto_scale_batch_size,
		'limit_train_batches': trainer.limit_train_batches,
		'model': trainer.model,
		'limit_test_batches':trainer.limit_test_batches
	}


def __safe_scale_batch_reset_params(trainer, model, steps_per_trial):
	trainer.auto_scale_batch_size = None  # prevent recursion
	trainer.auto_lr_find = False  # avoid lr find being called multiple times
	trainer.current_epoch = 0
	trainer.max_steps = steps_per_trial  # take few steps
	trainer.weights_summary = None  # not needed before full run
	trainer.logger = DummyLogger()
	trainer.callbacks = []  # not needed before full run
	trainer.limit_train_batches = 1.0
	trainer.optimizers, trainer.schedulers = [], []  # required for saving
	trainer.model = model  # required for saving
	trainer.limit_test_batches = steps_per_trial #test few steps


def __safe_scale_batch_restore_params(trainer):
	trainer.auto_lr_find = trainer.__dumped_params['auto_lr_find']
	trainer.current_epoch = trainer.__dumped_params['current_epoch']
	trainer.max_steps = trainer.__dumped_params['max_steps']
	trainer.weights_summary = trainer.__dumped_params['weights_summary']
	trainer.logger = trainer.__dumped_params['logger']
	trainer.callbacks = trainer.__dumped_params['callbacks']
	trainer.auto_scale_batch_size = trainer.__dumped_params['auto_scale_batch_size']
	trainer.limit_train_batches = trainer.__dumped_params['limit_train_batches']
	trainer.model = trainer.__dumped_params['model']
	trainer.limit_test_batches = trainer.__dumped_params['limit_test_batches']
	del trainer.__dumped_params
from pytorch_lightning.callbacks import ModelCheckpoint
from util.print_util import iprint

class NoVersionCheckpoint(ModelCheckpoint):
	'''
	Version of base Lightning checkpoint callback which has the option of overwriting checkpoints, so
	that I can debug the same experiment multiple times without having to keep deleting excess checkpoints.
	'''
	def __init__(self, *args, overwrite_existing=True, **kwargs):
		super(NoVersionCheckpoint, self).__init__(*args, **kwargs)
		self._overwrite_existing = overwrite_existing

	def _get_metric_interpolated_filepath_name(self, epoch, ckpt_name_metrics):
		filepath = self.format_checkpoint_name(epoch, ckpt_name_metrics)

		if not self._overwrite_existing:
			version_cnt = 0
			while self._fs.exists(filepath):
				filepath = self.format_checkpoint_name(
					epoch, ckpt_name_metrics, ver=version_cnt
				)
				# this epoch called before
				version_cnt += 1
		else:
			filepath = self.format_checkpoint_name(
				epoch, ckpt_name_metrics
			)
		return filepath

	def _save_model(self, filepath: str, trainer, pl_module):
		iprint(f'Saving checkpoint to {filepath}')
		super()._save_model(filepath, trainer, pl_module)
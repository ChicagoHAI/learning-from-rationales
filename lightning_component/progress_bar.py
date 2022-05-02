import sys

from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm


class ManualWidthProgressBar(ProgressBar):
	'''Version of the default lightning progress bar for which we can manually set the display width/ncols'''

	def __init__(self, *args, width=None, refresh_rate=10, **kwargs):
		super(ManualWidthProgressBar, self).__init__(*args, refresh_rate=refresh_rate, **kwargs)
		self._width = width

	def init_sanity_tqdm(self) -> tqdm:
		""" Override this to customize the tqdm bar for the validation sanity run. """
		bar = tqdm(
			desc='Validation sanity check',
			#position=0,
			position=(2 * self.process_position),
			disable=self.is_disabled,
			leave=False,
			dynamic_ncols=self._width is None,
			ncols=self._width,
			file=sys.stdout,
		)
		return bar

	def init_train_tqdm(self) -> tqdm:
		""" Override this to customize the tqdm bar for training. """
		bar = tqdm(
			desc='Training',
			initial=self.train_batch_idx,
			position=(2 * self.process_position),
			disable=self.is_disabled,
			leave=True,
			dynamic_ncols=self._width is None,
			ncols=self._width,
			file=sys.stdout,
			smoothing=0,
		)
		return bar

	def init_validation_tqdm(self) -> tqdm:
		""" Override this to customize the tqdm bar for validation. """
		bar = tqdm(
			desc='Validating',
			# position=0,
			position=(2 * self.process_position),
			# position=(2 * self.process_position),
			disable=self.is_disabled,
			leave=True,
			dynamic_ncols=self._width is None,
			ncols=self._width,
			file=sys.stdout
		)
		return bar

	def init_test_tqdm(self) -> tqdm:
		""" Override this to customize the tqdm bar for testing. """
		bar = tqdm(
			desc='Testing',
			position=(2 * self.process_position),
			disable=self.is_disabled,
			leave=True,
			dynamic_ncols=self._width is None,
			ncols=self._width,
			file=sys.stdout
		)
		return bar
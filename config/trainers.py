trainers = {
	'default_trainer': {
		'name': 'default_trainer',
		'params': {
			'min_epochs':1,
			'max_epochs': 2,
			'val_check_interval':.2,
			# 'precision': 16,
			# 'auto_scale_batch_size': 'binsearch',
			'auto_scale_safety_factor': 0.2,
			'num_sanity_val_steps': 0,
			'accumulate_grad_batches': 10,
			'log_every_n_steps':25

		}
	},
}

models = {

	'bert_rationale':{
		'class':'BertRationaleModel',
		'name':'bert_rationale',
		'params':{

			'gumbel_eval_temperature': 0.1,
			'gumbel_train_temperature': 0.5,
			'learning_rate': 2e-5,
		}
	},


	'bert_classification': {
		'class': 'BertClassificationModel',
		'name': 'bert_classification',
		'params': {
			'dropout_rate': 0.1,
			'learning_rate': 2e-5,
			'batch_size': 16,
			# 'warmup_steps':[50],
		}
	},
	'lstm_classification': {
		'class': 'LSTMClassificationModel',
		'name': 'lstm_classification',
		'params': {
			'dropout_rate': 0.1,
			'learning_rate': 1e-3,
			'batch_size': 64,
			# 'warmup_steps':[50],
		}
	},
	'logistic_regression': {
		'class': 'LogisticRegressionModel',
		'name': 'logistic_regression',
		'params': {
			'learning_rate': [1e-3],
			'batch_size': 16,
			# 'input_masking_strategy':['multiply_mask']
		}
	},

}

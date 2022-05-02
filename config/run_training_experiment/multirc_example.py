from config.datasets import datasets
from config.models import models
from config.trainers import trainers
# from config.full_models import models
from config.classes.script_config import TrainingScriptConfig
from config.global_config import global_config

config = {
	'script': TrainingScriptConfig(**{
		'output_dir': f'{global_config["output_directory"]}/run_experiment/multirc_example',
		'progress_bar_width': 250,
		# 'params':{'trial':[0,1,2]},
		'tune_batch_size_in_new_process': True,
		'train_model_in_new_process': False,
		'default_batch_size': 8, #use this to set a default batch size
		'do_training': True,
		'do_testing': True,
		'gpus': 1,
		'load_existing_model': False,
		'overwrite_existing_output': True,
		'ignore_nonexisting_model': False,
		'no_rationale_evaluation': False,
		'test_on_train_sample': False,
		'seed': 1234,
		'log_graph': False,
		'patience': 3,
	}),
	'read_dataset': {
		'cache_features_in_dataset_dir': True,
		'max_length': 512,
		'do_sentence_tokenization': True,
		'batch_width': 512,  # 'global_max', 'local_max'
	},
	'model_trainer': [


		# {
		# 	'model':models['bert_classification'],
		# 	'trainer':trainers['default_trainer'],
		# 	'model_params':{
		# 		# 'output_bert_attentions_on_test':True,
		# 		# 'output_simple_gradients_on_test':True,
		# 		# 'output_integrated_gradients_on_test':True,
		# 	}
		# },

		{
			'model': models['bert_rationale'],

			'trainer': trainers['default_trainer'],
			'trainer_params': {
				'accumulate_grad_batches': [10],
				'max_epochs': 1,
				'automatic_optimization': True,
				'val_check_interval': 0.2,
				'precision': 32,
				# 'limit_train_batches': 10,
				# 'limit_val_batches': 10,
				# 'limit_test_batches': 10,
			},
			'model_params': {
				'comment': 'Standalone unsupervised rationale model',
				'prediction_loss_weight': 1.0,
				'sparsity_loss_weight': [
					0.0
				],
				'human_rationale_loss_weight': [1.0],
				'provided_input_mask_method': 'assign_end',
				'predicted_rationale_masking_method': ['multiply_zero'],
				'predicted_rationale_binarization_method': ['gumbel_softmax'],
				'learning_rate': 2e-5,
				'rationalize_by_sentence':[True],
				# 'split_human_rationale_representation':[True],


			},
			'pretrain': [

				{
					'model': models['bert_classification'],
					'trainer': trainers['default_trainer'],
					'model_params': {
						# 'calibrate_with_masking':['random'],
						# 'calibration_parameter':[0.5]
						# 'train_with_human_input_masks':[True],
						# 'evaluate_with_human_input_masks':[True],
						# 'calibrate_with_masking':['human'],
						# 'input_masking_strategy':['multiply_zero']
					},
					'load_to': 'predictor'
				},
			],
		},




	],
	'dataset': [
		# datasets['two_class_good_bad_neutral'],
		# datasets['two_class_good_bad_neutral_query'],
		# datasets['two_class_good_bad_neutral_oneword_query'],
		# datasets['wiki_attack_rationales'],
		datasets['multirc'],
		# datasets['fever'],
		# datasets['esnli'],
		# datasets['movies'],
		# datasets['beeradvocate_taste_rationales_23k'],
		# datasets['beeradvocate_aroma_rationales_23k'],

	]
}

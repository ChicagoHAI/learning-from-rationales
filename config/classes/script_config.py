# from collections import Dict
from typing import List

class TrainingScriptConfig(dict):
	'''
	Just a
	'''

	def __init__(self,
				 output_dir: str, #output directory for experiment
				 # params:{trial:[0,1,2]},
				 tune_batch_size_in_new_process: bool, #whether to tune batch size in new process. Avoids gpu memory leaks
				 train_model_in_new_process: bool, #whether to do each training run in new process. Makes debugging harder but avoids aforementioned memory leaks.
				 default_batch_size: 16,
				 gpus: 1,
				 do_training: bool = True,
				 do_testing: bool = True,
				 load_existing_model: bool = False,
				 overwrite_existing_output: bool = True,
				 progress_bar_width: int = 200,
				 ignore_nonexisting_model: bool = False,
				 no_rationale_evaluation: bool = False,
				 is_pretraining: bool = False,
				shuffle_configs:bool=False,
				 test_on_train:bool=False,
				 **kwargs #todo get rid of this
				 ):
		super(TrainingScriptConfig, self).__init__(output_dir=output_dir,
												   tune_batch_size_in_new_process=tune_batch_size_in_new_process,
												   train_model_in_new_process=train_model_in_new_process,
												   default_batch_size=default_batch_size,
												   gpus=gpus,
												   do_training=do_training,
												   do_testing=do_testing,
												   load_existing_model=load_existing_model,
												   overwrite_existing_output=overwrite_existing_output,
												   progress_bar_width=progress_bar_width,
												   ignore_nonexisting_model=ignore_nonexisting_model,
												   no_rationale_evaluation=no_rationale_evaluation,
												   is_pretraining=is_pretraining,
												   shuffle_configs=shuffle_configs,
												   test_on_train=test_on_train,
												   **kwargs #todo get rid of this
												   )


class CrossEvaluationScriptConfig(dict):
	'''
	Just a
	'''

	def __init__(self,
				 output_dir: str, #output directory for experiment
				 evaluation_name: str,
				 # params:{trial:[0,1,2]},
				 tune_batch_size_in_new_process: bool,
				 evaluate_model_in_new_process: bool,
				 evaluation_input_masking_strategy: str,
				 default_batch_size: 16,
				 gpus: 1,
				 evaluate_with_test_set: bool = True,
				 evaluate_with_val_set: bool = True,
				 overwrite_existing_output: bool = True,
				 progress_bar_width: int = 200,
				 no_rationale_evaluation: bool=False,
				 no_same_model_different_param_crosses:bool=True,
				 shuffle_configs:bool=False,
				 test_random_rationales:bool=False,
				 random_rationale_probabilities:List=None
				 ):
		super(CrossEvaluationScriptConfig, self).__init__(output_dir=output_dir,
														  evaluation_name=evaluation_name,
														  tune_batch_size_in_new_process=tune_batch_size_in_new_process,
														  evaluate_model_in_new_process=evaluate_model_in_new_process,
														  default_batch_size=default_batch_size,
														  evaluation_input_masking_strategy=evaluation_input_masking_strategy,
														  gpus=gpus,
														  evaluate_with_test_set=evaluate_with_test_set,
														  evaluate_with_val_set=evaluate_with_val_set,
														  overwrite_existing_output=overwrite_existing_output,
														  progress_bar_width=progress_bar_width,
														  no_rationale_evaluation=no_rationale_evaluation,
														  no_same_model_different_param_crosses=no_same_model_different_param_crosses,
														  shuffle_configs=shuffle_configs,
														  test_random_rationales=test_random_rationales,
														  random_rationale_probabilities=random_rationale_probabilities
														  )

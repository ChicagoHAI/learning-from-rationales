from model.old.base_model import BaseModel
from transformers import BertModel, BertTokenizerFast
# from transformers.modeling_bert import BertModel
import torch
import torch.nn as nn
from model_component.masking import process_py_logits, mask_input, masked_mean, create_rationale_embedding_function
from torch.nn.functional import cross_entropy
from typing import Dict, List, Any, Union

from util.print_util import now, iprint
from torch.utils.data import Dataset
from util.dataloader import MaxSampler, RationaleDataloader
from transformers import get_constant_schedule_with_warmup
from model_component.gradients import generate_simple_gradients, generate_integrated_gradients
from model_component.evaluate_predictions import evaluate_predictions, evaluate_epoch
from model_component.evaluate_rationales import evaluate_rationale
from model_component.randomize import add_random_rationales, add_rationale_permutations

# from model.bert_rationale_model import BertRationaleModel

'''
Standard BERT classification model
'''

seed= 1234

class BertClassificationModel(BaseModel):

	def __init__(self,
				 *args,
				 output_bert_attentions_on_test: bool = None,
				 output_bert_attentions_on_eval: bool = False,
				 use_token_type_ids:bool=False,
				 train_with_empty_input_masks:bool=False,
				 evaluate_with_empty_input_masks:bool=False,
				 replace_all_document_tokens_with_mask:bool=False,
				 train_with_random_mask_chance:float=None,
				 **kwargs):
		super().__init__(*args, **kwargs)


		# todo compatibility thing. Get rid of this when I have new models.
		if output_bert_attentions_on_test is not None:
			self.output_bert_attentions_on_test = output_bert_attentions_on_test
		else:
			self.output_bert_attentions_on_test = output_bert_attentions_on_eval

		self.bert = BertModel.from_pretrained('bert-base-uncased',
											  output_attentions=self.output_bert_attentions_on_test)
		self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

		self.output_layer = nn.Linear(self.bert.config.hidden_size, self.num_classes)
		self.use_token_type_ids=use_token_type_ids
		self.replace_all_document_tokens_with_mask = replace_all_document_tokens_with_mask
		self.train_with_random_mask_chance = train_with_random_mask_chance

		# embedding_weight = torch.zeros((2, self.bert.config.hidden_size))
		# embedding_weight[0,-2]=1.0
		# embedding_weight[1,-1]=1.0
		# self.rationale_embedding_layer = torch.nn.Embedding(num_embeddings=2, embedding_dim= self.bert.config.hidden_size,_weight=embedding_weight)

		self.rationale_embedding_layer = create_rationale_embedding_function(self.bert, self.input_masking_strategy,self.bert.config.hidden_size)

		self.train_with_empty_input_masks = train_with_empty_input_masks
		self.evaluate_with_empty_input_masks =evaluate_with_empty_input_masks

	# self.dropout = nn.Dropout(self.dropout_rate)



	def forward(self,
				input_ids: torch.Tensor = None,
				padding_mask: torch.Tensor = None,
				label: torch.Tensor = None,
				human_rationale: torch.Tensor = None,
				human_rationale_weight: torch.Tensor = None,
				input_mask: torch.Tensor = None,
				return_embeddings: bool = False,
				special_mask: torch.Tensor = None,
				inputs_embeds: torch.Tensor = None,
				token_type_ids:torch.Tensor=None,
				**kwargs):

		result = {}
		# bert_result = self.bert(input_ids=input_ids, attention_mask=padding_mask)
		# pooled_output = bert_result[1]
		# result['py_logits'] = self.output_layer(pooled_output)
		# result['loss'] = cross_entropy(result['py_logits'], label, reduction='mean')
		#
		# return result

		#

		if self.replace_all_document_tokens_with_mask:
			input_ids = (self.tokenizer.mask_token_id*torch.ones_like(input_ids) * (1-special_mask) + input_ids * (special_mask)).int()

		if inputs_embeds is None:
			inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

		if (self.training and self.train_with_human_input_masks) or (self.eval and self.evaluate_with_human_input_masks):
			input_mask = human_rationale

		if (self.training and self.train_with_empty_input_masks) or (self.eval and self.evaluate_with_empty_input_masks):
			input_mask = torch.zeros_like(input_ids)

		if self.training and self.train_with_random_mask_chance is not None:
			rands = torch.rand(size=input_ids.size(),device=input_ids.device)
			input_mask = (rands < self.train_with_random_mask_chance).float()


		if input_mask is not None:

			# leave special tokens unmasked by default
			if special_mask is not None and self.use_special_mask:
				input_mask = 1 - ((1 - input_mask) * (1 - special_mask))

			result['input_mask'] = input_mask

			mask_result= mask_input(
				input_ids=input_ids,
				inputs_embeds=inputs_embeds,
				mask=input_mask,
				masking_strategy=self.input_masking_strategy,
				padding_mask=padding_mask,
				mask_token_id=self.tokenizer.mask_token_id,
				pad_token_id=self.tokenizer.pad_token_id,
				word_embedding_function=self.bert.embeddings.word_embeddings,
				rationale_embedding_function=self.rationale_embedding_layer)
			inputs_embeds_for_prediction, padding_mask_for_prediction = mask_result['masked_inputs_embeds'], mask_result['masked_padding_mask']
		else:
			inputs_embeds_for_prediction, padding_mask_for_prediction = inputs_embeds, padding_mask

		if return_embeddings: result['inputs_embeds'] = inputs_embeds

		token_type_ids_for_prediction = token_type_ids if self.use_token_type_ids else None
		bert_result = self.bert(inputs_embeds=inputs_embeds_for_prediction,
								token_type_ids=token_type_ids_for_prediction,
								attention_mask=padding_mask_for_prediction,
								)

		pooled_output = bert_result[1]
		result['py_logits'] = self.output_layer(pooled_output)
		result['py_index'], result['py_probs'] = process_py_logits(result['py_logits'])

		if label is not None:
			result['loss'] = cross_entropy(result['py_logits'], label, reduction='mean')

		if not self.training and self.forward_inputs_on_eval:
			result.update({
				'input_ids': input_ids,
				'padding_mask': padding_mask,
				'label': label,
				'human_rationale': human_rationale,
				'human_rationale_weight': human_rationale_weight
			})
			result.update(kwargs)



		if not self.training and self.output_bert_attentions_on_test and not self.suppress_own_rationales:
			bert_layer_attentions = bert_result[2]
			summed_bert_attention = torch.zeros_like(input_ids, dtype=torch.float32)
			for layer_num, bert_layer_attention in enumerate(bert_layer_attentions):
				summed_bert_attention += bert_layer_attention.sum(dim=2).sum(dim=1)
			# result[f'p_alpha_c_bert_{layer_num}'] =

			mean_sum = masked_mean(summed_bert_attention, mask=1-special_mask, dim=1).mean()
			summed_bert_attention = summed_bert_attention.masked_fill(special_mask.bool(), mean_sum)
			result['summed_bert_attention'] = summed_bert_attention

			result['logsum_bert_attention'] = torch.log(summed_bert_attention)


		# if self.suppress_own_rationales:
		# 		# 	result.pop('rationale')

		return result

	def embed(self, input_ids:torch.Tensor):
		return self.bert.embeddings.word_embeddings(input_ids)

	def training_step(self, batch, batch_idx):

		result = self.forward(**batch)
		loss = result['loss']
		self.log('train_loss', result['loss'])
		# self.evaluate_batch_or_epoch(batch, result, 'train', log_values=True)
		result.update(evaluate_predictions(model=self, batch=batch, result=result, log_values=True, prefix='train_'))

		# todo doing calibration this way is causing within-step memory leaks that are reducing the max batch size
		# in future, do this by having dataloader yield alternating normal batches and calibration batches
		# nah, just use the pytorch lightning multiple-optimizer infrastructure
		# and no AMP
		if self.calibrate_with_masking:
			calibration_result = self.run_calibration_batch(batch)
			loss += self.calibration_loss_weight * calibration_result['loss']

		return loss

	def load_from_pretrained(self, model: BaseModel, load_from: str=None, **kwargs):
		iprint(f'Loading parameters from pretrained model of class {model.__class__}')
		if model.__class__.__name__ == 'BertRationaleModel':
			iprint('Loading BertRationaleModel parameters into model')

			if 'predictor' == load_from:
				iprint('Loading BertRationaleModel predictor parameters into model')

				self.bert.load_state_dict(model.predictor.predictor.state_dict())
				self.output_layer.load_state_dict(model.predictor.output_layer.state_dict())

			elif 'generator' == load_from:
				iprint('Loading BertRationaleModel generator parameters into model')
				self.bert.load_state_dict(model.generator.generator.state_dict())
			else:
				raise Exception()

		elif model.__class__.__name__ == 'BertClassificationModel':

				iprint('Loading BertClassificationModel  parameters into model')
				self.bert.load_state_dict(model.bert.state_dict())
				self.output_layer.load_state_dict(model.output_layer.state_dict())


		else:
			raise Exception(f'Do not know how to load parameters from this class into own class {self.__class__}')

		return

	def initialize_dataloaders(self,
							   train_dataset: Dataset = None,
							   val_dataset: Dataset = None,
							   test_dataset: Dataset = None,
							   max_only: bool = False,
							   num_workers: int = None):
		'''
		Creates a self.train_dataloader() and self.val_dataloader() function which gets called by the trainer.

		We do this instead of manually specifying these things in the call to Trainer.fit() because the batch size
		tuner needs them to be there for it to do its thing.

		:param train_dataset:
		:param val_dataset:
		:param max_only: if true, initialize train_dataloader in such a way that it only returns the max-width-possible batch. Necessary when tuning batch size for variable-batch-width datasets, or tuner won't work right
		:param num_workers:
		:return:
		'''
		self.max_dataloaders = max_only
		if max_only:
			sampler = lambda d: MaxSampler(data_source=d)
		else:
			sampler = lambda d: None

		if num_workers is None: num_workers = self.num_workers

		if train_dataset is not None:
			self.train_dataloader = lambda: RationaleDataloader(
				train_dataset,
				description='internal_train',
				pad_token_id=self.tokenizer.pad_token_id,
				batch_size=self.batch_size,
				shuffle=self.shuffle_train and not max_only ,
				# shuffle=False,
				num_workers=num_workers,
				pin_memory=True,
				sampler=sampler(train_dataset)
				# multiprocessing_context='fork' if num_workers > 0 else None
			)


		if val_dataset is not None:
			self.val_dataloader = lambda: RationaleDataloader(
				val_dataset,
				description='internal_val',
				pad_token_id=self.tokenizer.pad_token_id,
				batch_size=self.batch_size,
				shuffle=False,  # todo see above
				num_workers=num_workers,
				pin_memory=True,
				sampler=sampler(val_dataset),  # only the training set needs to be max-only, practically speaking
				# multiprocessing_context='fork' if num_workers > 0 else None
			)

		if test_dataset is not None:
			self.test_dataloader = lambda: RationaleDataloader(
				test_dataset,
				description='internal_test',
				pad_token_id=self.tokenizer.pad_token_id,
				batch_size=self.batch_size,
				shuffle=False,  # todo see above
				num_workers=num_workers,
				pin_memory=True,
				sampler=sampler(test_dataset)
				# multiprocessing_context='fork' if num_workers > 0 else None
			)

		return

	def clear_dataloaders(self):
		self.train_dataloader = None

		self.val_dataloader = None

		self.test_dataloader = None


	def run_calibration_batch(self, batch):
		if self.calibrate_with_masking == 'human':
			calibration_mask = batch['human_rationale']
		elif self.calibrate_with_masking == 'random':
			probs = torch.ones_like(batch['input_ids']) * self.calibration_parameter
			calibration_mask = torch.bernoulli(probs)
		elif self.calibrate_with_masking == 'no_info':
			calibration_mask = torch.zeros_like(batch['input_ids'])
		elif self.calibrate_with_masking == 'full_info':
			calibration_mask = torch.ones_like(batch['input_ids'])
		calibration_result = self.forward(input_mask=calibration_mask, **batch)
		self.log('calibration_loss', calibration_result['loss'])
		return calibration_result

	def validation_step(self, batch, batch_idx):
		result = self.forward(**batch)
		self.log('val_loss', result['loss'])
		result.update(evaluate_predictions(model=self, batch=batch, result=result, log_values=True, prefix='val/'))
		return result

	def validation_epoch_end(self, outputs: List[Any]) -> None:
		'''
		At the end of the validation epoch, evaluate performance on the epoch and write predictions and
		evaluation to file
		:param outputs:
		:return:
		'''
		self.validation_epoch += 1
		evaluate_epoch(model=self, outputs=outputs, set='val', epoch=self.validation_epoch, no_file_io=self.no_file_io,write_predictions=False)

	def test_step(self, batch, batch_idx):
		'''
		We have the option of doing some computationally expensive operations during testing that we don't
		do during validation.

		One option is generating simple simple_gradients attribution mask based on output of forward()
		The other option is to evaluate the rationale(s) generated by forward(), including their fidelity

		:param batch:
		:param batch_idx:
		:return:
		'''

		result = self.forward(**batch)

		if self.suppress_human_rationales:
			result.pop('human_rationale')

		if self.output_simple_gradients_on_test and not self.suppress_own_rationales:
			# gradient_embeddings.requires_grad_(True)
			result['simple_gradients'] = generate_simple_gradients(model=self,
																   batch=batch)

		if self.output_integrated_gradients_on_test and not self.suppress_own_rationales:
			# We use mask token as baseline for integrated gradients because zero vector doesn't have an interpretation in BERT
			result['integrated_gradients'] = generate_integrated_gradients(model=self,
																		   batch=batch,
																		   # baseline_token_id = self.tokenizer.mask_token_id
																		   baseline_token_id = 'zero')

		for key in list(result.keys()):
			if 'rationale_full_info' in key or 'rationale_no_info' in key:
				result.pop(key)


		if self.test_edge_case_rationales:
			full_information = torch.ones_like(batch['input_ids'])
			result['rationale_full_info'] = full_information

			#Investigate what happens when the predictor or generator get no information. We can use this for fidelity normalization after the fact
			no_information = torch.zeros_like(batch['input_ids'])
			result['rationale_no_info'] = no_information

		if self.test_permuted_human_rationales:
			add_rationale_permutations(result, percentages=[0,.05, .1, .15, .2, .25, .3, .35, .4, .45, .5])
		# new rationales should be called e.g. "rationale_permuted_0.05"

		if self.test_random_rationales:
			add_random_rationales(result=result, probabilities=self.random_rationale_probabilities)

		result.update(evaluate_predictions(model=self, batch=batch, result=result, log_values=False, prefix=f'{self.test_output_prefix}/'))

		if not self.never_evaluate_rationales:
			rationales_evaluation_result = self.evaluate_rationales(batch=batch, original_result=result)
			result.update(rationales_evaluation_result)

		return result

	def test_epoch_end(self, outputs: List[Any]) -> None:
		# self.evaluate_at_epochs_end(outputs, self.test_output_prefix, self.validation_epoch)
		evaluate_epoch(model=self, outputs=outputs, set=self.test_output_prefix, epoch=-1, no_file_io = self.no_file_io, verbose=not self.no_evaluation_output)
		pass

	def evaluate_rationales(self, batch:Dict, original_result:Dict):
		'''
		For every output in the original_result dictionary that looks like it is a rationale/explanation, evaluate that output.

		For the purpose of fidelity, use the original result as the baseline. Note that this isn't appropriate for the RandomRationaleModel
		or BaseRationaleModel, so they need to implement their own versions of this method.
		:param batch:
		:param original_result:
		:return:
		'''

		#ignore any existing edge-case rationales passes from outside
		# for key in list(original_result.keys()):
		# 	if 'p_alpha_full_info' in key or 'p_alpha_no_info' in key:
		# 		original_result.pop(key)
		#
		# if self.test_edge_case_rationales:
		# 	full_information = torch.ones_like(batch['input_ids'])
		# 	original_result['rationale_full_info'] = full_information
		#
		# 	#Investigate what happens when the predictor or generator get no information. We can use this for fidelity normalization after the fact
		# 	no_information = torch.zeros_like(batch['input_ids'])
		# 	original_result['rationale_no_info'] = no_information

		rationale_keys = [key for key in original_result.keys() if self.is_rationale_key(key)]

		result = {}
		for key in rationale_keys:

			# if key == 'human_rationale': #todo get rid of this
			# 	iprint('Qrentlf')

			evaluation_result = evaluate_rationale(prefix=f'{self.test_output_prefix}/',
												   model=self,
												   rationale_name=key,
												   rationale_tensor=original_result[key],
												   batch=batch,
												   baseline_result=original_result,
												   evaluate_metrics=self.evaluate_rationales_on_test,
												   evaluate_fidelity=self.evaluate_fidelity_on_test,
												   num_quantiles=self.rationale_evaluation_quantiles)
			result.update(evaluation_result)
		return result

	def is_rationale_key(self,key):
		if key in ['human_rationale','predicted_rationale' ,'rationale_no_info','rationale_full_info']:
			return True
		elif 'permuted' in key and 'rationale' in key:
			return True
		elif 'attribution' in key:
			return True
		elif '_c_' in key:
			return True

		return False

	def get_progress_bar_dict(self):
		# don't show the version number
		items = super().get_progress_bar_dict()
		items.pop("v_num", None)
		return items

	def configure_optimizers(self):
		if self.warmup_steps is None:
			return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]
		else:
			iprint(f'Using linear LR warmup over {self.warmup_steps} steps')
			optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
			return [optimizer], [get_constant_schedule_with_warmup(optimizer, self.warmup_steps)]

	# def forward(self, *args, **kwargs):
	# 	raise NotImplementedError()



	def log_losses(self, result:Dict, prefix:str='', prog_bar:bool=False):
		for key,val in result.items():
			if key.endswith('loss'):
				self.log(prefix+key, val, prog_bar=prog_bar)

import pytorch_lightning as pl
from typing import Union, Dict, List, Any
import torch
from transformers import BertModel, BertTokenizerFast
from torch import nn as nn

from model_component.binarize import phi_to_rationale
from util.dataloader import MaxSampler, RationaleDataloader
from torch.utils.data import Dataset
from model_component.evaluate_predictions import evaluate_predictions, evaluate_epoch
from model_component.evaluate_rationales import evaluate_rationale
from model_component.masking import process_py_logits, masked_mean, mask_embeddings, create_rationale_embedding_function
from torch.nn.functional import cross_entropy, mse_loss
from transformers.models.bert.modeling_bert import BertModel
from model.bert_classification_model import BertClassificationModel
from util.print_util import iprint
from model_component.scatter import scatter_mean_gather_batch, scatter_mean_batch
# import torch_scatter as ts

# from transformers.modeling_bert import BertModel

class BertRationaleModel(pl.LightningModule):
	def __init__(self,
				 *args,
				 num_classes: int,
				 learning_rate: float,
				 output_dir: str = None,
				 dropout_rate: float = 0.1,
				 batch_size=10,
				 num_workers: Union[int, str] = 0,
				 warmup_steps: int = None,
				 provided_input_mask_method: str = 'removal',
				 note: str = None,
				 shuffle_train: bool = True,
				 train_with_human_input_masks: bool = False,
				 evaluate_with_human_input_masks: bool = False,
				 predicted_rationale_binarization_method: str,
				 predicted_rationale_masking_method: str,
				 prediction_loss_weight: float = 1.0,
				 sparsity_loss_weight: float = 0.0,
				 cohesiveness_loss_weight_multiple: float = 0.0,
				 human_rationale_loss_weight: float = 0.0,
				 human_rationale_loss_type: str = 'cross_entropy',
				 gumbel_eval_temperature: float = 0.1,
				 gumbel_train_temperature: float = 0.5,
				 separate_optimizers: bool = False,
				 human_rationale_recall_weight: float = None,
				 rationalize_by_sentence: bool = False,
				 comment: str = None,
				 **kwargs
				 ):

		super().__init__(*args, **kwargs)

		self.num_classes = num_classes
		self.learning_rate = learning_rate
		self.output_dir = output_dir
		self.dropout_rate = dropout_rate
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.warmup_steps = warmup_steps
		self.provided_input_mask_method = provided_input_mask_method
		self.note = note
		self.shuffle_train = shuffle_train
		self.train_with_human_input_masks = train_with_human_input_masks
		self.evaluate_with_human_input_masks = evaluate_with_human_input_masks
		self.predicted_rationale_binarization_method = predicted_rationale_binarization_method
		self.predicted_rationale_masking_method = predicted_rationale_masking_method
		self.prediction_loss_weight = prediction_loss_weight
		self.sparsity_loss_weight = sparsity_loss_weight
		self.cohesiveness_loss_weight_multiple = cohesiveness_loss_weight_multiple
		self.human_rationale_loss_weight = human_rationale_loss_weight
		self.human_rationale_loss_type = human_rationale_loss_type
		self.gumbel_eval_temperature = gumbel_eval_temperature
		self.gumbel_train_temperature = gumbel_train_temperature
		self.separate_optimizers = separate_optimizers
		self.rationalize_by_sentence = rationalize_by_sentence

		self.auto_metrics = {}
		self.validation_epoch = 0

		self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

		self.generator_bert = BertModel.from_pretrained("bert-base-uncased")
		self.generator_word_embedding_fn = lambda t: self.generator_bert.embeddings.word_embeddings(t)
		self.generator_output_size = self.generator_bert.config.hidden_size
		self.generator_output_layer = nn.Linear(self.generator_output_size, 1)
		self.generator_rationale_embedding_fn = create_rationale_embedding_function(bert_model=self.generator_bert,
																					masking_strategy=self.provided_input_mask_method,
																					embedding_size=self.generator_output_size)

		self.generator_parameters = list(self.generator_bert.parameters()) + list(self.generator_output_layer.parameters())

		self.predictor_bert = BertModel.from_pretrained("bert-base-uncased")
		self.predictor_word_embedding_fn = lambda t: self.predictor_bert.embeddings.word_embeddings(t)
		self.predictor_output_size = self.predictor_bert.config.hidden_size
		self.predictor_output_layer = nn.Linear(self.predictor_output_size, self.num_classes)
		self.predictor_rationale_embedding_fn = create_rationale_embedding_function(bert_model=self.predictor_bert,
																					masking_strategy=self.predicted_rationale_masking_method,
																					embedding_size=self.predictor_output_size)
		self.predictor_parameters = list(self.predictor_bert.parameters()) + list(self.predictor_output_layer.parameters())

		self.human_rationale_recall_weight = human_rationale_recall_weight

	def forward(self,
				input_ids: torch.Tensor,
				padding_mask: torch.Tensor = None,
				label: torch.Tensor = None,
				human_rationale: torch.Tensor = None,
				input_mask: torch.Tensor = None,
				special_mask: torch.Tensor = None,
				sentence_ids:torch.Tensor=None,
				**kwargs):

		result = {}

		#Generate predicted rationale
		generator_inputs_embeds = self.generator_word_embedding_fn(input_ids)
		generator_padding_mask = padding_mask

		if input_mask is not None and self.apply_input_mask_at == 'predictor':
			predicted_rationale = input_mask

		elif input_mask is None or self.apply_input_mask_at == 'generator':

			if input_mask is not None and self.apply_input_mask_at == 'generator':
				generator_mask_result = mask_embeddings(inputs_embeds=generator_inputs_embeds,
														mask=input_mask,
														padding_mask=padding_mask,
														word_embedding_function=self.generator_word_embedding_fn,
														mask_token_id=self.tokenizer.mask_token_id,
														rationale_embedding_function=self.generator_rationale_embedding_fn,
														masking_strategy=self.provided_input_mask_method
														)
				generator_inputs_embeds = generator_mask_result['masked_inputs_embeds']
				generator_padding_mask = generator_mask_result['masked_padding_mask']

			generator_output = self.generator_bert(inputs_embeds=generator_inputs_embeds,
												   attention_mask=generator_padding_mask)

			generator_hidden_states = generator_output['last_hidden_state']
			phi = self.generator_output_layer(generator_hidden_states).squeeze(2)

			if self.rationalize_by_sentence:
				sentence_phi = scatter_mean_batch(phi, sentence_ids)
				sentence_predicted_rationale = phi_to_rationale(
					phi=sentence_phi,
					binarization_method=self.predicted_rationale_binarization_method,
					training=self.training,
					gumbel_train_temperature=self.gumbel_train_temperature,
					gumbel_eval_temperature=self.gumbel_eval_temperature)


				predicted_rationale = torch.gather(sentence_predicted_rationale, 1, sentence_ids)
			else:

				predicted_rationale = phi_to_rationale(
					phi=phi,
					binarization_method=self.predicted_rationale_binarization_method,
					training=self.training,
					gumbel_train_temperature=self.gumbel_train_temperature,
					gumbel_eval_temperature=self.gumbel_eval_temperature)

		predicted_rationale = 1 - (1 - predicted_rationale) * (1 - special_mask)

		result['predicted_rationale'] = predicted_rationale

		#Generate rationalized prediction
		predictor_inputs_embeds = self.predictor_word_embedding_fn(input_ids)

		predictor_mask_result = mask_embeddings(inputs_embeds=predictor_inputs_embeds,
												mask=predicted_rationale,
												padding_mask=padding_mask,
												word_embedding_function=self.predictor_word_embedding_fn,
												mask_token_id=self.tokenizer.mask_token_id,
												rationale_embedding_function=self.predictor_rationale_embedding_fn,
												masking_strategy=self.predicted_rationale_masking_method
												)

		predictor_output = self.predictor_bert(inputs_embeds=predictor_mask_result['masked_inputs_embeds'],
											   attention_mask=predictor_mask_result['masked_padding_mask'])
		predictor_pooled_output = predictor_output['pooler_output']

		py_logits = result['py_logits'] = self.predictor_output_layer(predictor_pooled_output)

		result['py_index'], result['py_probs'] = process_py_logits(result['py_logits'])

		#Calculate most losses
		if label is not None:
			prediction_losses = cross_entropy(result['py_logits'], label, reduction='none')
			prediction_loss = result['prediction_loss'] = prediction_losses.mean()

		evaluation_mask = (1 - special_mask) * padding_mask  # where the rationale should be evaluated for sparsity and human rational accuracy. Ignore special tokens, the query, and padding tokens
		sparsity_loss = result['sparsity_loss'] = masked_mean(predicted_rationale, evaluation_mask).mean()

		if human_rationale is not None:
			predicted_rationale_probs = torch.sigmoid(phi)
			human_rationale_losses = mse_loss(predicted_rationale_probs, human_rationale, reduction='none')
			# human_rationale_losses = torch.nn.functional.binary_cross_entropy_with_logits(phi, human_rationale.float(), reduction='none')

			if self.human_rationale_recall_weight is not None:
				human_rationale_loss_weights = self.human_rationale_recall_weight * human_rationale + (1 - human_rationale)
				human_rationale_losses = human_rationale_losses * human_rationale_loss_weights

			human_rationale_loss = result['human_rationale_loss'] = masked_mean(human_rationale_losses, evaluation_mask, dim=1).mean()

		result['loss'] = self.prediction_loss_weight * prediction_loss + \
						 self.sparsity_loss_weight * sparsity_loss + \
						 self.human_rationale_loss_weight * human_rationale_loss

		result.update({
			'input_ids': input_ids,
			'padding_mask': padding_mask,
			'label': label,
			'human_rationale': human_rationale,
			# 'human_rationale_weight': human_rationale_weight
		})

		return result

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
				shuffle=self.shuffle_train and not max_only,
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

	def log_losses(self, result: Dict, prefix: str = '', prog_bar: bool = False):
		for key, val in result.items():
			if key.endswith('loss'):
				self.log(prefix + key, val, prog_bar=prog_bar)

	def configure_optimizers(self):
		if self.separate_optimizers:
			iprint('Using separate optimizers for generator and predictor')
			optimizers = torch.optim.Adam(self.generator_parameters, lr=self.learning_rate), torch.optim.Adam(self.predictor_parameters, lr=self.learning_rate)
		else:
			iprint('Using single optimizer for generator and predictor ')
			optimizers = [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]  # todo fix this
		if self.warmup_steps != None and self.warmup_steps != 0:
			raise Exception('No warmup right now')
		return optimizers

	def training_step(self, batch, batch_idx, optimizer_idx=None):

		if self.train_with_human_input_masks:
			batch['input_mask'] = batch['human_rationale']

		if not self.separate_optimizers:

			result = self.forward(**batch)

			evaluate_predictions(model=self, batch=batch, result=result, log_values=True, prefix='train/')
			self.log_losses(result, prefix='train_')

			return result
		else:
			if optimizer_idx == 0:
				self.freeze_predictor()
				result = self.forward(**batch)
				self.unfreeze_predictor()
			else:
				self.freeze_generator()
				result = self.forward(**batch)
				self.unfreeze_generator()
				evaluate_predictions(model=self, batch=batch, result=result, log_values=True, prefix='train/')
				self.log_losses(result, prefix='train_')

			return result

	def freeze_generator(self):
		for param in self.generator_parameters:
			param.requires_grad = False

	def unfreeze_generator(self):
		for param in self.generator_parameters:
			param.requires_grad = True

	def freeze_predictor(self):
		for param in self.predictor_parameters:
			param.requires_grad = False

	def unfreeze_predictor(self):
		for param in self.predictor_parameters:
			param.requires_grad = True

	def evaluation_step(self, batch, batch_idx, prefix=''):
		if self.evaluate_with_human_input_masks:
			result = self.forward(input_mask=batch['human_rationale'],
								  **batch)
		else:
			result = self.forward(**batch)

		evaluate_predictions(model=self, batch=batch, result=result, log_values=True, prefix=prefix + '/')
		self.log_losses(result, prefix=prefix + '_')

		rationale_evaluation_result = evaluate_rationale(prefix=prefix + '/',
														 model=self,
														 rationale_name='predicted_rationale',
														 rationale_tensor=result['predicted_rationale'],
														 batch=batch,
														 baseline_result=None,
														 evaluate_metrics=True,
														 evaluate_fidelity=False,
														 num_quantiles=None)
		result.update(rationale_evaluation_result)

		return result

	def validation_step(self, *args, **kwargs):
		return self.evaluation_step(*args, prefix='val', **kwargs)

	def validation_epoch_end(self, outputs: List[Any]) -> None:
		'''
		At the end of the validation epoch, evaluate performance on the epoch and write predictions and
		evaluation to file
		:param outputs:
		:return:
		'''
		self.validation_epoch += 1
		evaluate_epoch(model=self, outputs=outputs, set='val', epoch=self.validation_epoch, no_file_io=False, write_predictions=False)

	def test_step(self, *args, **kwargs):
		return self.evaluation_step(*args, prefix='test', **kwargs)

	def test_epoch_end(self, outputs: List[Any]) -> None:
		# self.evaluate_at_epochs_end(outputs, self.test_output_prefix, self.validation_epoch)
		evaluate_epoch(model=self, outputs=outputs, set=self.test_output_prefix, epoch=-1, no_file_io=False, verbose=True)
		pass

	def load_from_pretrained(self, model: pl.LightningModule, load_to: str):
		iprint(f'Loading parameters from pretrained model of class {model.__class__} into {load_to}')
		if isinstance(model, BertClassificationModel):# or isinstance(model, BertRationalePredictionModel):

			if 'predictor' in load_to:
				self.predictor_bert.load_state_dict(model.bert.state_dict())
				self.predictor_output_layer.load_state_dict(model.output_layer.state_dict())

			if 'generator' in load_to:
				self.generator_bert.load_state_dict(model.bert.state_dict())
				self.generator_output_layer.load_state_dict(model.output_layer.state_dict())

		# elif isinstance(model, BertRationaleModel):
		# 	self.predictor.load_state_dict(model.predictor.predictor.state_dict())
		# 	self.predictor_output_layer.load_state_dict(model.predictor.output_layer.state_dict())
		# 	self.generator.load_state_dict(model.generator.generator.state_dict())
		# 	self.generator_output_layer.load_state_dict(model.generator.generator_output_layer.state_dict())
		# elif isinstance(model, BaseBertRationaleComponents):
		# 	self.predictor.load_state_dict(model.predictor.state_dict())
		# 	self.generator.load_state_dict(model.generator.state_dict())
		# 	self.predictor_output_layer.load_state_dict(model.predictor_output_layer.state_dict())
		# 	self.generator_output_layer.load_state_dict(model.generator_output_layer.state_dict())

		else:
			raise Exception(f'Do not know how to load parameters from this class into own class {self.__class__}')

		return

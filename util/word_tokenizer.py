import torch.nn as nn
from config.pretrained import pretrained
import numpy as np
from util.print_util import iprint
import os
import pickle
from util.misc_util import ensure_dir_exists
from typing import List, Union
from transformers import BertTokenizerFast
from nltk.tokenize import WordPunctTokenizer
from pprint import pformat

seed=1234

class WordTokenizer():
	'''
	Has minimal amount of the functionality of BertTokenizerFast to do conventional tokenization
	and token ID assignment based on GloVe.

	Intended for use with LSTM the same way BertTokenizerFast is used with Bert
	'''

	def __init__(self,
				 embedding_name = 'glove_300',
				 ):
		self.embeddings = GloveEmbeddings(filepath=pretrained[embedding_name]['path'], 
										  add_tokens = ['[PAD]','[SEP]','[CLS]','[UNK]','[MASK]'],
										  seed=seed)
		self.tokenizer = WordPunctTokenizer()
		
		self.sep_token = '[SEP]'
		self.sep_token_id = self.embeddings.token_dict[self.sep_token]

		self.pad_token = '[PAD]'
		self.pad_token_id = self.embeddings.token_dict[self.pad_token]

		#Probably don't need this, but keeping it just in case
		self.cls_token = '[CLS]'
		self.cls_token_id = self.embeddings.token_dict[self.cls_token]

		self.mask_token = '[MASK]'
		self.mask_token_id = self.embeddings.token_dict[self.mask_token]

		self.unk_token = '[UNK]'
		self.unk_token_id = self.embeddings.token_dict[self.unk_token]


	def convert_ids_to_tokens(self, ids:List[int]):
		return [self.embeddings.tokens[id] for id in ids]

	def convert_tokens_to_ids(self, tokens:List[str]):
		return [self.embeddings.token_dict[token] if token in self.embeddings.token_dict else self.unk_token_id for token in tokens]

	def encode_plus(self,
					text:str=None,
					text_pair:str=None,
					return_offsets_mapping:bool=True,
					add_special_tokens:bool=True,
					max_length:int=None,
					truncation:Union[bool,str]=False,
					return_token_type_ids=True,
					return_special_tokens_mask=True):

		text = text.lower()
		token_spans = list(self.tokenizer.span_tokenize(text))
		tokens = [text[span[0]:span[1]] for span in token_spans]

		if text_pair is not None:
			text_pair = text_pair.lower()
			pair_spans = list(self.tokenizer.span_tokenize(text_pair))
			pair_tokens = [text_pair[span[0]:span[1]] for span in pair_spans]

			offset = len(text)
			pair_spans = [(span[0]+offset, span[1]+offset) for span in pair_spans]

			# if add_special_tokens:
			# 	#Since this isn't BERT, we don't add a [CLS] at the beginning or a [SEP] at the end, only a [SEP] separating the document and query
			# 	pair_tokens.insert(0, self.sep_token)
			# 	pair_spans.insert(0, None)
		else:
			pair_tokens = []
			pair_spans =[]


		if add_special_tokens and len(pair_tokens) > 0:
			#Since this isn't BERT, we don't add a [CLS] at the beginning or a [SEP] at the end, only a [SEP] separating the document and query
			tokens.append(self.sep_token)
			token_spans.append((0,0))

		if max_length is not None and (len(tokens) + len(pair_tokens)) > max_length:

			# Mimic default BertTokenizer truncation strategy, which is to truncate the longer of the two sequences
			if truncation == True:
				if len(tokens) >= len(pair_tokens):
					tokens = tokens[0:max_length-len(pair_tokens)]
					token_spans = token_spans[0:max_length-len(pair_tokens)]
				else:
					pair_tokens = pair_tokens[0:max_length-len(tokens)]
					pair_spans = pair_spans[0:max_length-len(tokens)]

			elif type(truncation) == 'str':
				raise NotImplementedError('Truncation strategies not implemented')
			elif truncation == False:
				raise Exception('Sequence exceeded max length')



		rdict = {'input_ids':self.convert_tokens_to_ids(tokens+pair_tokens)}
		if return_token_type_ids:
			rdict['token_type_ids'] = [0 for token in tokens] + [1 for token in pair_tokens]

		if return_special_tokens_mask:
			rdict['special_tokens_mask'] = [1 if token == self.sep_token else 0 for token in tokens+pair_tokens]

		if return_offsets_mapping:
			rdict['offset_mapping'] = token_spans+pair_spans

		#Not bothing with token_type_ids or attention_mask for now, since I don't use either of those downstream
		return rdict

class GloveEmbeddings():
	def __init__(self,
				 filepath:str=None,
				 add_tokens = None,
				 seed=None):
		self.tokens, self.token_dict, self.embeddings = read_embedding_file(filepath,add_tokens=add_tokens, seed=seed)


def read_embedding_file(filepath:str, use_cache=True, add_tokens=None, seed=None):

	iprint('Reading embedding file')
	if use_cache:
		cache_dir = os.path.join(os.path.dirname(filepath),'_'.join(os.path.basename(filepath).split('.')[:-1])+'_cache')
		iprint(f'Looking for cache in {cache_dir}')

		tokens_path = os.path.join(cache_dir, 'tokens.pkl')
		token_dict_path = os.path.join(cache_dir, 'token_dict.pkl')
		embeddings_path = os.path.join(cache_dir, 'embeddings.npy')
		if all([os.path.exists(path) for path in [tokens_path,token_dict_path,embeddings_path]]):
			iprint(f'Loading cache')
			with open(tokens_path, 'rb') as f:
				tokens = pickle.load(f)
			with open(token_dict_path, 'rb') as f:
				token_dict = pickle.load(f)
			with open(embeddings_path, 'rb') as f:
				embeddings = np.load(f)
			cache_hit = True
		else:
			iprint('No cache found')
			cache_hit = False

	if not use_cache or not cache_hit:
		tokens = []
		vectors = []
		iprint(f'Loading embeddings from {filepath}')
		with open(filepath, 'r') as f:
			for line in f:
				pieces = line.split(' ')
				tokens.append(pieces[0])
				vectors.append([float(piece) for piece in pieces[1:]])

		if seed is not None: np.random.seed(seed)
		if add_tokens is not None:
			iprint(f'Adding a random embedding for each of the following: {add_tokens}')
			for add_token in add_tokens:
				tokens.append(add_token)
				vectors.append(np.random.rand(len(vectors[0])))
		token_dict = {token:index for index, token in enumerate(tokens)}
		embeddings = np.array(vectors)
		iprint(f'{len(tokens)} embeddings loaded.')


	if use_cache and not cache_hit:
		iprint(f'Caching embeddings to {cache_dir}')
		ensure_dir_exists(cache_dir)
		with open(tokens_path, 'wb') as f:
			pickle.dump(tokens,f)
		with open(token_dict_path, 'wb') as f:
			pickle.dump(token_dict,f)
		with open(embeddings_path, 'wb') as f:
			np.save(f, embeddings)

	return tokens, token_dict, embeddings

def main():
	combined = 'I am an archmage.'+'Am I an archmage?'

	btokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
	bencoded = btokenizer.encode_plus(text='I am an archmage.',
									text_pair ='Am I an archmage?',
									return_offsets_mapping=True,
									add_special_tokens=True)

	iprint(pformat(bencoded))
	iprint(btokenizer.convert_ids_to_tokens(bencoded['input_ids']))
	iprint(' '.join([combined[span[0]:span[1]] for span in bencoded['offset_mapping'] if span is not None]))

	tokenizer = WordTokenizer()
	encoded = tokenizer.encode_plus(text='I am an archmage.',
									text_pair ='Am I an archmage?',
									return_offsets_mapping=True,
									add_special_tokens=True)

	iprint(pformat(encoded))
	iprint(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
	iprint(' '.join([combined[span[0]:span[1]] for span in encoded['offset_mapping'] if span is not None]))

	pass


if __name__ == '__main__':
	main()

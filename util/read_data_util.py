import pandas as pd
from util.print_util import iprint
from typing import List, Union
import os
from util.dataset import RationaleDataset
import numpy as np
from nltk import PunktSentenceTokenizer

from util.pseudoexample_util import create_pseudoexamples_from_df


def read_dataset(filepath: str,
				 tokenizer:object,
				 classes: List,
				 cache_directory: str = None,
				 cache_features_in_dataset_dir=False,
				 max_length:int=None,
				 query_special_if_unspecified:bool=True,
				 do_sentence_tokenization:bool=True,
				 add_pseudoexamples:bool=False,
				 pseudoexample_type:str=None,
				 pseudoexample_proportion:float=0.0,
				 pseudoexample_parameter:float=0.0,
				 return_token_spans:float=False,
				 batch_width:Union[str, int]=None
				 ):
	'''
	This function reads the data, tokenizes it, combines documents and queries as appropriate, optionally caches the features,
	and returns a dataframe and a pytorch Dataset for it

	This function expects the data to be in the form of line-separated json records with the following fields:
	id: unique identifier
	document: text of document
	document_rationale_spans (optional): spans of document text corresponding to ground truth rationale values, if available
	document_rationale_values (optional): values of ground truth rationale, if available

	query (optional): text of query, if present
	query_rationale_spans (optional): spans of query text corresponding to ground truth rationale values, if available
	query_rationale_values (optional): values of ground truth rationale, if available

	It produces the following values for each row:
	input_ids: token ids for document and query, appended together with special tokens added
	rationale: ground truth rationale if available
	rationale_weight: binary indicator of which ground-truth rationale values should be learned from or evaluated on. 0s out special tokens and usually the query.
	label: index of true class

	:param filepath: dataset file path
	:param tokenizer: BertTokenizer or equivalent
	:param classes: list of class labels
	:param cache_directory: where to put cache
	:param cache_features_in_dataset_dir: whether to put cache in dataset directory
	:param max_length: max sequence length
	:return:
	'''

	iprint(f'Reading dataset from {filepath}')
	df = pd.read_json(filepath, orient='records', lines=True)
	df['text_length'] = df['document'].apply(len) + df['query'].apply(len)

	if cache_features_in_dataset_dir:
		if cache_directory is not None: raise Exception('Conflicting cache location choices')
		cache_directory = os.path.join(os.path.dirname(filepath), 'feature_cache', f'{str(tokenizer.__class__.__name__)}')

	# Try to load features from cache if available. Validate them if successful
	if cache_directory is not None:

		cache_path = os.path.join(cache_directory, os.path.basename(filepath))
		if os.path.exists(cache_path):
			iprint(f'Found feature cache at {cache_path}')
			cache_df = pd.read_json(cache_path, orient='records', lines=True)
			cache_df['text_length'] = cache_df['document'].apply(len) + cache_df['query'].apply(len)
			cache_first_200 = cache_df.sort_values('text_length',ascending=False).head(200).reset_index(drop=True).drop(columns='text_length')
			cache_df.drop(columns='text_length',inplace=True)

			# Make sure the cache is correct
			tokenized_first_200 = tokenize_df(df.sort_values('text_length',ascending=False).head(200).reset_index(drop=True),
											  tokenizer,
											  classes,
											  concat=True,
											  max_length=max_length,
											  query_special_if_unspecified=query_special_if_unspecified,
											  do_sentence_tokenization=do_sentence_tokenization,
											  return_token_spans=return_token_spans).drop(columns='text_length')
			df.drop(columns='text_length',inplace=True)

			if (tokenized_first_200.equals(cache_first_200)):
				iprint('Cache matches!')
				tokenized_df = cache_df
				# cache_df.drop(columns=['text_length'], inplace=True)
			else:
				iprint('ERROR: Cache does not match current tokenization code. Re-tokenizing and caching.')
				tokenized_df = tokenize_df(df,
										   tokenizer,
										   classes,
										   max_length=max_length,
										   query_special_if_unspecified=query_special_if_unspecified,
										   do_sentence_tokenization=do_sentence_tokenization,
										   return_token_spans=return_token_spans)
				write_df_cache(tokenized_df, cache_path)
		else:
			iprint(f'No feature cache at {cache_path}, so tokenizing and caching')
			tokenized_df = tokenize_df(df,
									   tokenizer,
									   classes,
									   max_length=max_length,
									   query_special_if_unspecified=query_special_if_unspecified,
									   do_sentence_tokenization=do_sentence_tokenization,
									   return_token_spans=return_token_spans)
			write_df_cache(tokenized_df, cache_path)
	else:
		tokenized_df = tokenize_df(df,
								   tokenizer,
								   classes,
								   max_length=max_length,
								   query_special_if_unspecified=query_special_if_unspecified,
								   do_sentence_tokenization=do_sentence_tokenization,
								   return_token_spans=return_token_spans)

	tokenized_df['sequence_length'] = tokenized_df['token_ids'].apply(len)
	iprint(f'{tokenized_df.shape[0]} examples loaded')
	iprint(f'Mean sequence length: {tokenized_df["sequence_length"].mean():.2f}')


	# if tile:
	# 	iprint(f'Tiling dataset {tile} times for benchmarking')
	# 	tokenized_df=pd.DataFrame(np.repeat(tokenized_df.values,tile,axis=0),columns=tokenized_df.columns)
	# 	iprint(f'{tokenized_df.shape[0]} instances in tiled dataset')

	if add_pseudoexamples:
		iprint(f'Generating {pseudoexample_proportion} "{pseudoexample_type}"-type pseudoexamples per original example')

		if cache_directory is not None:
			pseudoexample_cache_directory = os.path.join(cache_directory, 'pseudoexamples',f'{pseudoexample_type}_{pseudoexample_proportion}_{pseudoexample_parameter}')
			pseudoexample_cache_path = os.path.join(pseudoexample_cache_directory, os.path.basename(filepath))
			if os.path.exists(pseudoexample_cache_path):
				iprint(f'Found pseudoexample cache at {pseudoexample_cache_path}')
				cache_pseudoexample_df = pd.read_json(pseudoexample_cache_path, orient='records', lines=True)

				# Make sure the cache is correct
				first_100_pseudoexamples = create_pseudoexamples_from_df(df=tokenized_df,
																		  pseudoexample_type=pseudoexample_type,
																		  pseudoexample_proportion=pseudoexample_proportion,
																		 pseudoexample_parameter=pseudoexample_parameter,
																		  max_pseudoexamples=100)
				if (first_100_pseudoexamples.equals(cache_pseudoexample_df.head(100))):
					iprint('Cached pseudoexamples are a match, so using them.')
					pseudoexample_df = cache_pseudoexample_df
				else:
					iprint('ERROR: Pseudoexample cache does not match current pseudoexample code. Regenerating pseudoexamples')
					pseudoexample_df = create_pseudoexamples_from_df(df=tokenized_df,
																	 pseudoexample_type=pseudoexample_type,
																	 pseudoexample_proportion=pseudoexample_proportion,
																	 pseudoexample_parameter=pseudoexample_parameter)
					write_df_cache(pseudoexample_df, pseudoexample_cache_path)
			else:
				iprint(f'No pseudoexample cache at {pseudoexample_cache_path}, so tokenizing and caching')
				pseudoexample_df = create_pseudoexamples_from_df(df=tokenized_df,
																 pseudoexample_type=pseudoexample_type,
																 pseudoexample_proportion=pseudoexample_proportion,
																 pseudoexample_parameter=pseudoexample_parameter)
				write_df_cache(pseudoexample_df, pseudoexample_cache_path)
		else:
			pseudoexample_df = create_pseudoexamples_from_df(df=tokenized_df,
															 pseudoexample_type=pseudoexample_type,
															 pseudoexample_proportion=pseudoexample_proportion,
															 pseudoexample_parameter=pseudoexample_parameter)
		iprint(f'{pseudoexample_df.shape[0]} pseudoexamples generated to add to {tokenized_df.shape[0]} real examples')

		tokenized_df = pd.concat([tokenized_df,pseudoexample_df],axis=0).reset_index(drop=True)

	iprint(f'Class balance:\n{tokenized_df["label"].value_counts()/tokenized_df.shape[0]}')

	dataset = RationaleDataset(
		input_ids=tokenized_df['token_ids'],
		human_rationale=tokenized_df['human_rationale'],
		human_rationale_weight=tokenized_df['human_rationale_weight'],
		special_mask=tokenized_df['special_mask'],
		label=tokenized_df['y'],
		sentence_ids = tokenized_df['sentence_ids'] if do_sentence_tokenization else None,
		pseudoexample = tokenized_df['pseudoexample'] if add_pseudoexamples else None,
		token_type_ids = tokenized_df['token_type_ids'],
		batch_width=batch_width
	)


	return tokenized_df, dataset


def write_df_cache(df, cache_path):
	try:
		os.makedirs(os.path.dirname(cache_path))
	except:
		pass
	df.to_json(cache_path, orient='records', lines=True)

def tokenize_df(df,
				tokenizer,
				classes: List,
				concat=True,
				max_length:int=None,
				query_special_if_unspecified:bool=True,
				do_sentence_tokenization:bool=False,
				return_token_spans:bool=False):
	if 'label' in df:
		df['y'] = df['label'].apply(lambda l: classes.index(l))
	else:
		df['y'] = None
	tokenizations = []

	if do_sentence_tokenization:
		sentence_tokenizer = PunktSentenceTokenizer()
	else:
		sentence_tokenizer = None

	for i, row in df.iterrows():
		tokenization = tokenize_with_rationale(tokenizer, **row,
											   max_length=max_length,
											   query_special_if_unspecified=query_special_if_unspecified,
											   sentence_tokenizer=sentence_tokenizer,
											   return_token_spans=return_token_spans)
		tokenizations.append(tokenization)

	tokenized_df = pd.DataFrame(tokenizations)
	if concat: tokenized_df = pd.concat([df, tokenized_df], axis=1)
	return tokenized_df


def is_valid_example(df_row):
	'''
	Make sure document exists and is not just whitespace
	:param df_row:
	:return:
	'''
	document = df_row['document']
	if pd.isnull(document):
		return False
	elif document.strip() == '':
		return False
	else:
		return True

def tokenize_with_rationale(tokenizer,
							document: str,
							document_rationale_spans: List = None,
							document_rationale_values: List = None,
							query: str = None,
							query_rationale_spans: List = None,
							query_rationale_values: List = None,
							max_length:str=None,
							query_special_if_unspecified:bool=True,
							sentence_tokenizer:PunktSentenceTokenizer=None,
							return_token_spans:bool=False,
							**kwargs):
	'''
	Take in a document and optionally a query and produce a set of tokens. If human rationales are present,
	align them with the tokenization we're performing
	:param tokenizer:
	:param document:
	:param document_rationale_spans:
	:param document_rationale_values:
	:param query:
	:param query_rationale_spans:
	:param query_rationale_values:
	:param query_special_if_unspecified: include query in the special mask if it's not specified via query_rationale_spans/query_rationale_values
	:param kwargs:
	:return:
	'''
	tokenization = {}

	encoded = tokenizer.encode_plus(text=document,
									text_pair=query,
									return_offsets_mapping=True,
									add_special_tokens=True,
									max_length=max_length,
									truncation=True,
									return_token_type_ids=True,
									return_special_tokens_mask=True)
	tokenization['special_mask'] = encoded['special_tokens_mask']
	if query_special_if_unspecified and np.any(pd.isnull(query_rationale_values)):
		tokenization['special_mask'] = [int(t1 or t2) for t1, t2 in zip(tokenization['special_mask'], encoded['token_type_ids'])]

	tokenization['token_ids'] = encoded['input_ids']
	tokenization['tokens'] = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
	tokenization['human_rationale'], tokenization['human_rationale_weight'] = compute_token_rationale(
		tokenization_spans=encoded['offset_mapping'],
		document=document,
		document_rationale_spans=document_rationale_spans,
		document_rationale_values=document_rationale_values,
		query=query,
		query_rationale_spans=query_rationale_spans,
		query_rationale_values=query_rationale_values,

	)
	if return_token_spans:
		# the offset mapping is a list of tuples, but it gets saved to JSON as a list of lists. So we need to do this for the cache to work.
		tokenization['tokenization_spans'] = [list(span) for span in encoded['offset_mapping']]
	tokenization['token_type_ids'] = encoded['token_type_ids']

	assert len(tokenization['token_ids']) == len(tokenization['human_rationale'])
	assert len(tokenization['human_rationale']) == len(tokenization['human_rationale_weight'])


	if max_length is not None:
		assert len(tokenization['token_ids']) <= max_length

	if sentence_tokenizer is not None:
		tokenization['sentence_ids'] = compute_sentence_ids(document=document,
															query=query,
															tokenization_mapping=encoded['offset_mapping'],
															sentence_tokenizer=sentence_tokenizer)


	return tokenization


def compute_sentence_ids(document:str, tokenization_mapping:List, sentence_tokenizer:PunktSentenceTokenizer, query:str=None, return_sentence_spans=False):
	'''
	Use the document, the query (if present), the offset mapping from the word or wordpiece tokenization, and the NLTK sentence tokenizer,
	to assign a sentence ID to each token.


	:param document:
	:param tokenization_mapping: list of tuples with token offsets for document and query. special tokens are (0,0), and query offsets are NOT offset relative to the document,
	so we need to be a little careful about how we do the alignment
	:param query:
	:return: list of sentence IDs, one per token in the tokenization mapping. Set all special tokens to be sentence -1
	'''

	#todo there is a bug here where the tokenization mapping is based on a version of the doc and query where they have been truncated
	# to be <= 512 tokens total.
	# Solution: truncate the output of sentence_tokenizer accordingly.

	section_tokens = split_list(tokenization_mapping, (0,0))
	
	text=document

	document_tokens = section_tokens[0]
	max_document_offset = document_tokens[-1][1]
	document_sentences = list(sentence_tokenizer.span_tokenize(document))
	document_sentences = [sentence for sentence in document_sentences if sentence[0] < max_document_offset]

	
	if query is not None:
		text+=query
		query_tokens = section_tokens[1]
		max_query_offset = query_tokens[-1][1]
		query_sentences = list(sentence_tokenizer.span_tokenize(query))
		query_sentences = [sentence for sentence in query_sentences if sentence[0] < max_query_offset]
		sentences = document_sentences + query_sentences
	else:
		sentences = document_sentences

	sentence_ids = []
	current_sentence_id = 0

	for token_num,token_offset in enumerate(tokenization_mapping):
		if token_offset == (0,0):
			sentence_ids.append(0)
			current_sentence_id += 1
		else:
			if (not span_partial_contains(sentences[current_sentence_id-1], token_offset) >= 0.5) and current_sentence_id < len(sentences):
				current_sentence_id += 1
				assert span_partial_contains(sentences[current_sentence_id-1], token_offset) >= 0.5, f"Token {token_offset} wasn't contained in {sentences[current_sentence_id-2]} and should be contained in {sentences[current_sentence_id-1]}, but isn't..."
			sentence_ids.append(current_sentence_id)

	assert max(sentence_ids) == len(sentences), f'Max sentence ID {max(sentence_ids)} was not equal to final sentence {len(sentences)}'

	if return_sentence_spans:
		return sentence_ids, sentences
	else:
		return sentence_ids

from typing import Any
def split_list(l:List, val:Any):
	lists = []
	current_list = []
	for i, item in enumerate(l):
		if item == val:
			if len(current_list) > 0:
				lists.append(current_list)
				current_list = []
		else:
			current_list.append(item)

	if len(current_list) > 0:
		lists.append(current_list)

	return lists




def test_compute_sentence_ids():
	iprint('Testing compute_sentence_ids()')
	document = 'You are a bum. Go to hell.'
	document_tokens = [(0,3), (4,7), (8,9), (10,13), (13,14), (15,17), (18,20), (21,25), (25,26)]
	query = 'Are you a bum? Perhaps.'
	query_tokens = [(0,3), (4,7), (8,9), (10,13),(13,14), (15,22),(22,23)]
	tokenization_mapping = [(0,0)]+document_tokens+[(0,0)]+query_tokens+[(0,0)]

	print(document)
	print([document[token[0]:token[1]] for token in document_tokens])
	print(query)
	print([query[token[0]:token[1]] for token in query_tokens])

	sentence_tokenizer= PunktSentenceTokenizer()

	true_sentence_spans = [(0,14),(15,26),(0,14),(15,23)]
	true_sentence_ids = [-1, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, 2, 2, 2, 2, 2, 3, 3, -1]

	sentence_ids,sentence_spans = compute_sentence_ids(document=document, query=query, tokenization_mapping=tokenization_mapping, sentence_tokenizer=sentence_tokenizer, return_sentence_spans=True)

	assert sentence_spans == true_sentence_spans, f'Computed sentence spans \n{sentence_spans}\ndoes not match true sentence spans\n{true_sentence_spans}'
	assert sentence_ids == true_sentence_ids, f'Computed sentence IDs \n{sentence_ids}\ndoes not match true sentence IDs\n{true_sentence_ids}'
	print('Testing complete. All tests passed')


def compute_token_rationale(
		tokenization_spans: List,
		document: str = None,
		document_rationale_spans: List = None,
		document_rationale_values: List[float] = None,
		query: str = None,
		query_rationale_spans: List = None,
		query_rationale_values: List[float] = None,
		default_value=1
):
	'''
	Figure out what the combined ground truth rationale should be for the document and query and their associated set of rationale spans and values.

	Chief complication is to align the rationale spans provided by the dataset with the ones generated by the tokenizer. If they are irreconcilably misaligned,
	consider retiring to a remote mountain range and becoming a Buddhist monk.

	:param tokenization_spans: spans of tokens generated by the tokenizer. Span will be None for special tokens like [SEP] that were inserted by the tokenizer
	:param document: doucment text
	:param document_rationale_spans: spans of document tokens associated with ground-truth rationale
	:param document_rationale_values: values of ground-truth rationale
	:param query: query text
	:param query_rationale_spans: spans of query tokens associated with ground-truth rationale, if present
	:param query_rationale_values: values of ground-truth query rationale, if present
	:return:
	'''

	# combined_text = document
	combined_rationale_spans = []
	combined_rationale_values = []

	if document_rationale_spans is not None:
		combined_rationale_spans.extend(document_rationale_spans)
		combined_rationale_values.extend(document_rationale_values)

	if query_rationale_spans is not None and not np.any(pd.isnull(query_rationale_spans)):
		# document_offset = len(document)
		# combined_rationale_spans.extend([[span[0] + document_offset, span[1] + document_offset] for span in query_rationale_spans])
		combined_rationale_spans.extend(query_rationale_spans)
		combined_rationale_values.extend(query_rationale_values)

	combined_span_num = 0
	token_rationale = []
	token_rationale_weights = []  # So we can disregard special tokens
	for span_num, span in enumerate(tokenization_spans):

		if span is None or span == (0,0): #Indicates that this span was associated with a special token added by the tokenizer like [SEP]
			token_rationale.append(default_value)
			token_rationale_weights.append(0.0)
		else:
			while combined_span_num < len(combined_rationale_spans) and not span_contains(combined_rationale_spans[combined_span_num], span):
				combined_span_num += 1

			if combined_span_num < len(combined_rationale_spans):
				token_rationale.append(combined_rationale_values[combined_span_num])
				token_rationale_weights.append(1.0)
			else:
				token_rationale.append(default_value)
				token_rationale_weights.append(0.0)

	return token_rationale, token_rationale_weights


def span_contains(container: List, contained: List):
	if contained[0] >= container[0] and contained[1] <= container[1]:
		return True
	else:
		return False


def span_partial_contains(container: List, contained: List):

	contained_span = contained[1]- contained[0]
	portion_contained = max(0, contained_span - ( max(0, container[0]-contained[0]) + max(0,contained[1]-container[1])))/contained_span

	return portion_contained

def test_partial_contains():
	print('Testing partial containment')
	assert span_partial_contains([60, 100], [50,55]) == 0.0
	assert span_partial_contains([60, 100], [55,65]) == 0.5
	assert span_partial_contains([60, 100], [65,70]) == 1.0
	assert span_partial_contains([60, 100], [95,110]) == 1/3
	print('All tests passed')


def test_compute_token_rationale():
	iprint('Testing compute_token_rationale() function')
	document = 'there is a bucket'
	document_rationale_spans = [[0, 5], [6, 8], [9, 10], [11, 17]]
	assert ' '.join([document[span[0]:span[1]] for span in document_rationale_spans]) == document
	document_rationale_values = [0, 0, 1, 1]

	query = 'there is no damn bucket'
	query_rationale_spans = [[0, 5], [6, 8], [9, 11], [12, 16], [17, 23]]
	assert ' '.join([query[span[0]:span[1]] for span in query_rationale_spans]) == query
	query_rationale_values = [0, 0, 1, 1, 1]

	combined_text = document + query
	tokenized_text = '[CLS] there is a buck ##et [SEP] there is no buck ##et [SEP]'
	tokenization_spans = [None, [0, 5], [6, 8], [9, 10], [11, 15], [15, 17], None, [0 + 17, 5 + 17], [6 + 17, 8 + 17], [9 + 17, 11 + 17], [12 + 17, 16 + 17], [17 + 17, 21 + 17], [21 + 17, 23 + 17], None]

	# assert ' '.join([combined_text[span[0]:span[1]] for span in tokenization_spans if span is not None]) == combined_text

	token_rationale_1, token_rationale_weights_1 = compute_token_rationale(
		tokenization_spans=tokenization_spans,
		document=document,
		document_rationale_spans=document_rationale_spans,
		document_rationale_values=document_rationale_values,
		query=query,
		query_rationale_spans=query_rationale_spans,
		query_rationale_values=query_rationale_values
	)

	assert token_rationale_1 == [-1, 0, 0, 1, 1, 1, -1, 0, 0, 1, 1, 1, 1, -1]
	assert token_rationale_weights_1 == [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]

	token_rationale_2, token_rationale_weights_2 = compute_token_rationale(
		tokenization_spans=tokenization_spans,
		document=document,
		document_rationale_spans=document_rationale_spans,
		document_rationale_values=document_rationale_values,
		query=query,
		query_rationale_spans=None,
		query_rationale_values=None)

	assert token_rationale_2 == [-1, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
	assert token_rationale_weights_2 == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

	token_rationale_3, token_rationale_weights_3 = compute_token_rationale(
		tokenization_spans=tokenization_spans,
		document=document,
		document_rationale_spans=None,
		document_rationale_values=None,
		query=query,
		query_rationale_spans=None,
		query_rationale_values=None)

	assert token_rationale_3 == [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
	assert token_rationale_weights_3 == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	iprint('Tests passed!')



def main():
	# test_compute_token_rationale()
	# test_partial_contains()
	test_compute_sentence_ids()


if __name__ == '__main__':
	main()

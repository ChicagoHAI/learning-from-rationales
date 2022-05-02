import numpy as np
import pandas as pd
import scipy as sp

seed = 29929

'''
Various functions for creating new training examples for variant training regimes
'''

def create_pseudoexamples_from_df(df:pd.DataFrame, pseudoexample_type:str, pseudoexample_proportion:float, pseudoexample_parameter:float, max_pseudoexamples:int=None):
	df['pseudoexample']=False
	pseudoexample_rows = []
	np.random.seed(seed)

	num_pseudoexamples = 0
	num_pseudoexamples_to_generate = int(df.shape[0] * pseudoexample_proportion)
	if max_pseudoexamples is not None:
		num_pseudoexamples_to_generate = min(num_pseudoexamples_to_generate, max_pseudoexamples)

	dataset_repeats = int(np.ceil(pseudoexample_proportion))

	for dataset_repeat in range(dataset_repeats):
		shuffled_indices = df.index.values.copy()
		np.random.shuffle(shuffled_indices)
		for shuffled_index,(row_index, row) in zip(shuffled_indices, df.iterrows()):
			pseudoexample_row = create_pseudoexample(example_row=row, swap_row = df.loc[shuffled_index], pseudoexample_type=pseudoexample_type, pseudoexample_parameter=pseudoexample_parameter)
			pseudoexample_rows.append(pseudoexample_row)
			num_pseudoexamples += 1
			if num_pseudoexamples >= num_pseudoexamples_to_generate:
				break

		if num_pseudoexamples >= num_pseudoexamples_to_generate:
			break
	pseudoexample_df = pd.DataFrame(pseudoexample_rows)
	# combined_df = pd.concat([df, swap_df]).reset_index(drop=True)
	return pseudoexample_df


def create_pseudoexample(example_row:pd.Series, swap_row:pd.Series, pseudoexample_type:str, pseudoexample_parameter:float):
	if pseudoexample_type == 'within_dataset_sentence_swap':
		pseudoexample_row = swap_sentences_between_two_rows(row=example_row, swap_row=swap_row)
	elif pseudoexample_type == 'within_example_sentence_replace':
		pseudoexample_row = replace_irrelevant_sentences(row=example_row, replace_proportion=pseudoexample_parameter)
	elif pseudoexample_type == 'within_example_sentence_shuffle_replace':
		pseudoexample_row = replace_irrelevant_sentences(row=example_row, replace_proportion=pseudoexample_parameter, shuffle_sentences=True)
	elif pseudoexample_type == 'within_example_sentence_drop':
		pseudoexample_row = drop_irrelevant_sentences(row=example_row, drop_proportion=pseudoexample_parameter)
	elif pseudoexample_type == 'human_rationale_masking':
		pseudoexample_row = add_input_mask(row=example_row, mask_key = 'rationale')
	else:
		raise Exception(f'Unknown sentence-swapping strategy "{pseudoexample_type}"')

	required_keys = ['special_mask', 'token_ids', 'tokens', 'rationale', 'rationale_weight', 'token_type_ids', 'sentence_ids']
	for key in required_keys:
		assert key in pseudoexample_row, f'Required key "{key}" is missing from "{pseudoexample_type}" pseudoexample'
	assert pseudoexample_row['pseudoexample'] == True

	return pseudoexample_row

def add_input_mask(row:pd.Series, mask_key:str):
	new_row = row.copy()
	new_row['pseudoexample'] = True
	new_row['input_mask'] = new_row[mask_key]
	return new_row



def drop_irrelevant_sentences(row:pd.Series, drop_proportion:float, max_len:int=512):
	'''
	Remove some percentage of non-rationale sentences from the row and shift everything to accommodate that removal.

	Intuition here is to create pseudoexamples where irrelevant data has been dropped, but which still are fairly
	IID relative to the actual data

	:param row:
	:param drop_proportion:
	:param max_len:
	:return:
	'''
	sentence_spans = calculate_sentence_token_spans(row['sentence_ids'])
	rationale_sentences = np.array([True  if 1.0 in row['rationale'][span[0]:span[1]] else False for span in sentence_spans])
	nonrationale_sentence_indices = np.nonzero(1-rationale_sentences)[0]
	np.random.shuffle(nonrationale_sentence_indices)
	num_replacement_sentences = int(drop_proportion * len(nonrationale_sentence_indices))
	remove_sentence_indices = nonrationale_sentence_indices[:num_replacement_sentences]
	sequence_keys = ['special_mask', 'token_ids', 'tokens', 'rationale', 'rationale_weight', 'token_type_ids', 'sentence_ids']
	new_row = {key:[] for key in sequence_keys}
	for i in range(len(sentence_spans)):
		if not rationale_sentences[i]:
			if i in remove_sentence_indices:
				continue
			else:
				span = sentence_spans[i]
		else:
			span = sentence_spans[i]

		for key in sequence_keys:
			new_row[key].extend(row[key][span[0]:span[1]])

	new_row['pseudoexample'] = True
	new_row['y'] = row['y']
	new_row['original_id'] = row['id']
	new_row['sequence_length'] = len(new_row['tokens'])

	# compare_text(row, pseudoexample=new_row)

	return new_row


def replace_irrelevant_sentences(row:pd.Series, replace_proportion:float, max_len:int=512, shuffle_sentences:bool=False):
	'''
	For a certain sample of non-rationale sentences, replace those sentences with repetitions of other nonrationale sentences
	from the non-replacement sample. Also shuffle all sentences around if desired

	Intuition here is to remove irrelevant data without changing the length or composition of the example

	:param row:
	:param replace_proportion:
	:param max_len:
	:param shuffle_sentences:
	:return:
	'''
	sentence_spans = calculate_sentence_token_spans(row['sentence_ids'])
	if shuffle_sentences:
		sep_token_index = row['tokens'].index('[SEP]')
		sep_sentence_index = [i for i,span in enumerate(sentence_spans) if span[0] <= sep_token_index and span[1] >= sep_token_index][0]
		document_sentences = sentence_spans[1:sep_sentence_index]
		query_sentences = sentence_spans[sep_sentence_index:]
		np.random.shuffle(document_sentences)
		sentence_spans =sentence_spans[0:1] + document_sentences + query_sentences

	rationale_sentences = np.array([True  if 1.0 in row['rationale'][span[0]:span[1]] else False for span in sentence_spans])
	num_row_rationale_tokens = sum([span[1]-span[0] for span, is_rationale in zip(sentence_spans, rationale_sentences) if is_rationale])
	replace_token_budget = max_len-num_row_rationale_tokens

	nonrationale_sentence_indices = np.nonzero(1-rationale_sentences)[0]
	np.random.shuffle(nonrationale_sentence_indices)
	num_replacement_sentences = int(replace_proportion * len(nonrationale_sentence_indices))
	replace_sentence_indices = nonrationale_sentence_indices[:num_replacement_sentences]
	nonreplace_sentence_indices = nonrationale_sentence_indices[num_replacement_sentences:]

	sequence_keys = ['special_mask', 'token_ids', 'tokens', 'rationale', 'rationale_weight', 'token_type_ids', 'sentence_ids']
	new_row = {key:[] for key in sequence_keys}

	replace_token_count = 0
	for i in range(len(sentence_spans)):
		if not rationale_sentences[i]:
			if i in replace_sentence_indices:
				span = sentence_spans[nonreplace_sentence_indices[i % len(nonreplace_sentence_indices)]]
			else:
				span = sentence_spans[i]
			replace_token_count += span[1]-span[0]
			if replace_token_count > replace_token_budget:
				continue
		else:
			span = sentence_spans[i]

		for key in sequence_keys:
			new_row[key].extend(row[key][span[0]:span[1]])

	new_row['pseudoexample'] = True
	new_row['y'] = row['y']
	new_row['original_id'] = row['id']
	new_row['sequence_length'] = len(new_row['tokens'])

	# compare_text(row, pseudoexample=new_row)

	return new_row

# def sample_with_replacement_and_rollover(a, n):
# 	if n > a.shape[0]:
# 		a = np.tile(a, n // a.shape[0])
#
# 	return np.random.choice(a, n)

def swap_sentences_between_two_rows(row:pd.Series, swap_row:pd.Series, max_len:int=512):
	'''
	Replace non-rationale sentences with corresponding sentences from other examples in the dataset

	Idea is to create pseudoexamples where the relevant data is preserved, which are still IID
	relative to the real data

	Example row:
		id                                 Fiction-stories-masc-A_Wasted_Day-2.txt:0:0
		label                                                                    False
		document                     As his car slid downtown on Tuesday morning th...
		document_rationale_spans     [[0, 2], [3, 6], [7, 10], [11, 15], [16, 24], ...
		document_rationale_values    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
		query                        How does Mr. Thorndike act upon his impulse ? ...
		query_rationale_spans                                                      NaN
		query_rationale_values                                                     NaN
		y                                                                            0
		special_mask                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
		token_ids                    [101, 2004, 2010, 2482, 4934, 5116, 2006, 9857...
		tokens                       [[CLS], as, his, car, slid, downtown, on, tues...
		rationale                    [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0....
		rationale_weight             [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...
		token_type_ids               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
		sentence_ids                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
		sequence_length                                                            454
		sentence_swapped                                                         False

	Everything below y needs to be changed by the swap. The stuff above it should be replaced with NaNs to avoid confusing the sentence swapped row with the original

	:param df_row:
	:param swap_row:
	:return:
	'''

	row_sentence_spans = calculate_sentence_token_spans(row['sentence_ids'])
	row_rationale_sentences = [True  if 1.0 in row['rationale'][span[0]:span[1]] else False for span in row_sentence_spans]
	num_row_rationale_tokens = sum([span[1]-span[0] for span, is_rationale in zip(row_sentence_spans, row_rationale_sentences) if is_rationale])
	swap_budget = max_len-num_row_rationale_tokens


	swap_row_sentence_spans = calculate_sentence_token_spans(swap_row['sentence_ids'])


	swap_row_sentence_token_types = [max(swap_row['token_type_ids'][span[0]:span[1]]) for span in swap_row_sentence_spans]
	swap_row_sentence_special_tokens = [max(swap_row['special_mask'][span[0]:span[1]]) for span in swap_row_sentence_spans]


	sequence_keys = ['special_mask', 'token_ids', 'tokens', 'token_type_ids', 'sentence_ids']

	# sequence_keys = ['special_mask', 'token_ids', 'tokens', 'rationale', 'rationale_weight', 'token_type_ids', 'sentence_ids']
	new_row = {key:[] for key in sequence_keys}
	new_row.update({'rationale':[], 'rationale_weight':[]})
	swap_count = 0
	for i in range(len(row_sentence_spans)):

		if not row_rationale_sentences[i]:

			if i >= len(swap_row_sentence_spans) or swap_row_sentence_token_types[i] == 1 or swap_row_sentence_special_tokens[i] == 1:
				continue

			contributor = swap_row
			span = swap_row_sentence_spans[i]
			swap_count += span[1]-span[0]
			if swap_count > swap_budget:
				continue

			new_row['rationale'].extend([0]*(span[1]-span[0]))
			new_row['rationale_weight'].extend([1.0]*(span[1]-span[0]))
		else:
			contributor = row
			span = row_sentence_spans[i]
			new_row['rationale'].extend(contributor['rationale'][span[0]:span[1]])
			new_row['rationale_weight'].extend(contributor['rationale_weight'][span[0]:span[1]])


		for key in sequence_keys:
			new_row[key].extend(contributor[key][span[0]:span[1]])

	new_row['pseudoexample'] = True
	new_row['y'] = row['y']
	new_row['original_id'] = row['id']
	new_row['swap_id'] = swap_row['id']
	new_row['sequence_length'] = len(new_row['tokens'])

	assert new_row['sequence_length'] <= 512

	return new_row

	# compare_text(row, swap_row, sequences)


def calculate_sentence_token_spans(sentence_ids):
	spans = []
	current_span = [0,1]
	current_id = sentence_ids[0]
	sentence_type_ids = []
	sentence_type_id = 0
	for i in range(1, len(sentence_ids)):

		if sentence_ids[i] == current_id:
			current_span[1] += 1
		else:
			spans.append(current_span)
			current_span= [current_span[1], current_span[1]+1]
		current_id= sentence_ids[i]

	spans.append(current_span)


	return spans

def compare_text(row, swap_row=None, pseudoexample=None):

	print('\nOriginal text:')
	print(' '.join(row['tokens']))

	print('\nOriginal text (rationalized):')
	print(' '.join([token if ri == 1 else '-'*len(token) for token, ri in zip(row['tokens'], row['rationale'])]))

	if swap_row is not None:
		print('\nSwap row text:')
		print(' '.join(swap_row['tokens']))


	print('\nPseudoexample text:')
	print(' '.join(pseudoexample['tokens']))

	print('\nPseudoexample text (rationalized):')
	print(' '.join([token if ri == 1 else '-'*len(token) for token, ri in zip(pseudoexample['tokens'], pseudoexample['rationale'])]))

def test_calculate_sentence_token_spans():
	print('Testing calculate_sentence_token_spans()')
	ids_1 = [0, 1, 1, 1, 2,2,0,3,3,3,3,0]
	spans_1 = [[0,1],[1,4],[4,6],[6,7],[7,11],[11,12]]
	predicted_spans_1 = calculate_sentence_token_spans(ids_1)
	assert predicted_spans_1 == spans_1, 'Test failed'

	print('All tests succeeded')


if __name__ == '__main__':
	test_calculate_sentence_token_spans()
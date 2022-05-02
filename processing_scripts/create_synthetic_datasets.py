from util.print_util import iprint
from util.misc_util import dump_json, tokens_to_spans
import os
import pandas as pd
import numpy as np

np.set_printoptions(linewidth=1000, precision=3, suppress=True, threshold=3000)
pd.set_option('display.width', 3000)
pd.set_option('max_colwidth', 50)
pd.set_option('max_columns', 20)
pd.set_option('precision', 3)

'''
Create one or more simple synthetic datasets we can use to test classification and rationalization
'''
configs = [
	# {
	# 	'output_dir': '../data/synthetic/two_class_good_bad_neutral',
	# 	'name': 'Two-class good-bad-neutral vocabulary',
	# 	'num_examples': [10000, 2000, 2000],
	# 	'class_balance': [0.5, 0.5],
	# 	'classes': ['good', 'bad'],
	# 	'text_function': 'text_from_class_and_cpt',
	# 	'cpt': {'good': {'good': {'good': 0.45, 'neutral': 0.5, '[END]': 0.05},
	# 					 'neutral': {'good': 0.1, 'neutral': 0.85, '[END]': 0.05}},
	# 			'bad': {'bad': {'bad': 0.45, 'neutral': 0.5, '[END]': 0.05},
	# 					'neutral': {'bad': 0.1, 'neutral': 0.85, '[END]': 0.05}}},
	# 	'seed': 38848,
	# 	'max_text_length': 512
	# },
	{
		'output_dir': '../data/synthetic/two_class_good_bad_neutral_query',
		'name': 'Two-class good-bad-neutral vocabulary with a query',
		'num_examples': [10000, 2000, 2000],
		'class_balance': [0.5, 0.5],
		'classes': ['good', 'bad'],
		'text_function': 'text_from_class_and_cpt',
		'cpt': {'good': {'good': {'good': 0.45, 'neutral': 0.5, '[END]': 0.05},
						 'neutral': {'good': 0.1, 'neutral': 0.85, '[END]': 0.05}},
				'bad': {'bad': {'bad': 0.45, 'neutral': 0.5, '[END]': 0.05},
						'neutral': {'bad': 0.1, 'neutral': 0.85, '[END]': 0.05}}},
		'seed': 38848,
		'max_text_length': 512,
		'query': 'what is the valence ?',
	},
	# {
	# 	'output_dir': '../data/synthetic/two_class_good_bad_neutral_oneword',
	# 	'name': 'Two-class good-bad-neutral vocabulary with one function word per text',
	# 	'num_examples': [10000, 2000, 2000],
	# 	'class_balance': [0.5, 0.5],
	# 	'classes': ['good', 'bad'],
	# 	'text_function': 'text_from_class_and_cpt',
	# 	'cpt': {'good': {'good': {'good': 0.0, 'neutral': 0.95, '[END]': 0.05},
	# 					 'neutral': {'good': 0.0, 'neutral': 0.95, '[END]': 0.05}},
	# 			'bad': {'bad': {'bad': 0.0, 'neutral': 0.95, '[END]': 0.05},
	# 					'neutral': {'bad': 0.0, 'neutral': 0.95, '[END]': 0.05}}},
	# 	'seed': 38848,
	# 	'max_text_length': 512
	# },
	# {
	# 	'output_dir': '../data/synthetic/two_class_good_bad_neutral_oneword_query',
	# 	'name': 'Two-class good-bad-neutral vocabulary with one function word per text and a query',
	# 	'num_examples': [10000, 2000, 2000],
	# 	'class_balance': [0.5, 0.5],
	# 	'classes': ['good', 'bad'],
	# 	'text_function': 'text_from_class_and_cpt',
	# 	'cpt': {'good': {'good': {'good': 0.0, 'neutral': 0.95, '[END]': 0.05},
	# 					 'neutral': {'good': 0.0, 'neutral': 0.95, '[END]': 0.05}},
	# 			'bad': {'bad': {'bad': 0.0, 'neutral': 0.95, '[END]': 0.05},
	# 					'neutral': {'bad': 0.0, 'neutral': 0.95, '[END]': 0.05}}},
	# 	'seed': 38848,
	# 	'max_text_length': 512,
	# 	'query': 'what is the valence ?',
	# },
	# {
	# 	'output_dir': '../data/synthetic/two_class_good_bad_neutral_oneword_query_0.0',
	# 	'name': 'Two-class good-bad-neutral vocabulary with one functional word, with a constant query with target rationale 0',
	# 	'num_examples': [10000, 2000, 2000],
	# 	'class_balance': [0.5, 0.5],
	# 	'classes': ['good', 'bad'],
	# 	'text_function': 'text_from_class_and_cpt',
	# 	'cpt': {'good': {'good': {'good': 0.0, 'neutral': 0.90, '[END]': 0.1},
	# 					 'neutral': {'good': 0.0, 'neutral': 0.90, '[END]': 0.1}},
	# 			'bad': {'bad': {'bad': 0.0, 'neutral': 0.90, '[END]': 0.1},
	# 					'neutral': {'bad': 0.0, 'neutral': 0.90, '[END]': 0.1}}},
	# 	'seed': 38848,
	# 	'max_text_length': 512,
	# 	'query': 'what is the valence ?',
	# 	'query_rationale_value': 0.0
	# },
	# {
	# 	'output_dir': '../data/synthetic/short_two_class_good_bad_neutral_oneword_query_0.0',
	# 	'name': 'Very short text two-class good-bad-neutral vocabulary with one functional word, with a constant query with target rationale 0',
	# 	'num_examples': [10000, 2000, 2000],
	# 	'class_balance': [0.5, 0.5],
	# 	'classes': ['good', 'bad'],
	# 	'text_function': 'text_from_class_and_cpt',
	# 	'cpt': {'good': {'good': {'good': 0.0, 'neutral': 0.80, '[END]': 0.2},
	# 					 'neutral': {'good': 0.0, 'neutral': 0.80, '[END]': 0.2}},
	# 			'bad': {'bad': {'bad': 0.0, 'neutral': 0.80, '[END]': 0.2},
	# 					'neutral': {'bad': 0.0, 'neutral': 0.80, '[END]': 0.2}}},
	# 	'seed': 38848,
	# 	'max_text_length': 50,
	# 	'query': 'what is the valence ?',
	# 	'query_rationale_value': 0.0
	# },
	# {
	# 	'output_dir': '../data/synthetic/long_two_class_good_bad_neutral_oneword_query_0.0',
	# 	'name': 'Long text two-class good-bad-neutral vocabulary with one functional word, with a constant query with target rationale 0',
	# 	'num_examples': [10000, 2000, 2000],
	# 	'class_balance': [0.5, 0.5],
	# 	'classes': ['good', 'bad'],
	# 	'text_function': 'text_from_class_and_cpt',
	# 	'cpt': {'good': {'good': {'good': 0.0, 'neutral': 0.999, '[END]': 0.001},
	# 					 'neutral': {'good': 0.0, 'neutral': 0.999, '[END]': 0.001}},
	# 			'bad': {'bad': {'bad': 0.0, 'neutral': 0.999, '[END]': 0.001},
	# 					'neutral': {'bad': 0.0, 'neutral': 0.999, '[END]': 0.001}}},
	# 	'seed': 38848,
	# 	'max_text_length': 512,
	# 	'query': 'what is the valence ?',
	# 	'query_rationale_value': 0.0
	# }

]


def main():
	for config in configs:
		iprint('Generating synthetic data according to following config:')
		iprint(config)

		if not os.path.isdir(config['output_dir']):
			os.makedirs(config['output_dir'])

		dump_json(config, os.path.join(config['output_dir'], 'config.json'))

		if config['text_function'] == 'text_from_class_and_cpt':
			text_function = text_from_class_and_cpt
		else:
			raise Exception(f'Unknown function {config["text_function"]}')

		np.random.seed(config['seed'])

		for set, set_size in zip(['train', 'dev', 'test'], config['num_examples']):
			iprint(set)
			examples = []
			classes = np.random.choice(config['classes'], size=set_size, p=config['class_balance'])
			interval = len(classes) // 20

			for example_num, example_class in enumerate(classes):
				if (example_num  % interval) == 0:
					iprint(f"{example_num + 1}/{len(classes)}...")
				example_text, example_rationale_spans, example_rationale_values = text_function(example_class, config['cpt'][example_class], config['classes'], max_length=config['max_text_length'])
				example = {'id': example_num,
						   'label': example_class,
						   'document': example_text,
						   'document_rationale_spans': example_rationale_spans,
						   'document_rationale_values': example_rationale_values}

				if 'query' in config:
					example['query'] = config['query']

				if 'query_rationale_value' in config:
					example['query_rationale_spans'] = tokens_to_spans(config['query'].split(' '))
					example['query_rationale_values'] = [config['query_rationale_value'] for span in example['query_rationale_spans']]
				examples.append(example)

			set_df = pd.DataFrame(examples)

			iprint(f"Mean text length: {set_df['document'].apply(lambda t: len(t.split(' '))).mean()}")
			iprint(f"Mean rationale value: {set_df['document_rationale_values'].apply(np.mean).mean()}")
			iprint(f"Class counts: {set_df['label'].value_counts()}")

			set_path = os.path.join(config['output_dir'], f'{set}.json')
			iprint(set_path)
			set_df.to_json(set_path, orient='records', lines=True)
		iprint('Done!')


def text_from_class_and_cpt(start_token, cpt, rationale_tokens, max_length=None, add_query=False):
	'''
	Generate text by starting with the given token and then proceeding in both directions according to the given
	conditional probability table
	:param text_class:
	:param cpt:
	:return:
	'''
	current_tokens = [start_token, start_token]
	sequence = [start_token]
	attach_functions = [lambda token:sequence.insert(0,token), lambda token: sequence.append(token)]

	while len(current_tokens) > 0 and (len(sequence) < max_length or max_length is None):
		current_token = current_tokens.pop(0)
		attach_function = attach_functions.pop(0)

		next_token = sample_from_cpt(current_token, cpt)
		if next_token == '[END]':
			pass
		else:
			attach_function(next_token)
			current_tokens.append(next_token)
			attach_functions.append(attach_function)

	full_sequence = sequence
		# sequences.append(sequence)
	# full_sequence = sequences[1][:0:-1] + sequences[0]


	# if max_length is not None and len(full_sequence) > max_length:
	#
	# 	begin_fraction = (len(sequences[1])-1)/len(full_sequence)
	# 	end_fraction = (len(sequences[0])-1)/len(full_sequence)
	#
	# 	remove = len(full_sequence) - max_length
	# 	begin_remove = int(round(remove * begin_fraction))
	# 	end_remove = int(round(remove * end_fraction))
	# 	while (begin_remove + end_remove) < remove:
	# 		if begin_remove > end_remove:
	# 			begin_remove += 1
	# 		else:
	# 			end_remove += 1
	# 	original_sequence = full_sequence
	# 	full_sequence = full_sequence[begin_remove:len(full_sequence)-end_remove]

	for token in full_sequence:
		assert token in ['neutral', start_token]
	assert start_token in full_sequence
	assert len(full_sequence) <= max_length or max_length is None

	spans = tokens_to_spans(full_sequence, sep=' ')
	rationale_spans_values = [(span, 1) if token in rationale_tokens else (span, 0) for span, token in zip(spans, full_sequence)]
	rationale_spans, rationale_values = zip(*rationale_spans_values)

	# rationale = [1 if token in rationale_tokens else 0 for token in full_sequence]
	text = ' '.join(full_sequence)
	return text, rationale_spans, rationale_values


def sample_from_cpt(token, cpt):
	tokens, probs = zip(*cpt[token].items())
	new_token = np.random.choice(tokens, p=probs)
	return new_token


if __name__ == '__main__':
	main()

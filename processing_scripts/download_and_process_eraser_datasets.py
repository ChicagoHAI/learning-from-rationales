from util.print_util import iprint
from util.misc_util import set_display_options, tokens_to_spans

set_display_options()
import os
import subprocess
import json
import re
import pandas as pd
from pprint import pformat, pprint
import numpy as np
import sys
from util.display_util import sample_and_output_as_html
from typing import List, Dict
from config.global_config import global_config


'''
Download and process datasets included in the ERASER repository

'''

# todo add citations
seed=8388
eraser_dir = f'{global_config["data_directory"]}/eraser'
eraser_datasets = [
	# { #check
	# 	'name':'esnli',
	# 	'url':'https://www.eraserbenchmark.com/zipped/esnli.tar.gz',
	#  },
	{ #check
		'name':'multirc',
		'url':'https://www.eraserbenchmark.com/zipped/multirc.tar.gz',
	},
	# { #check
	# 	'name':'fever',
	# 	'url':'https://www.eraserbenchmark.com/zipped/fever.tar.gz',
	# },
	# { #check
	# 	'name':'boolq',
	# 	'url':'https://www.eraserbenchmark.com/zipped/boolq.tar.gz',
	# },
	# { #check
	# 	'name':'cose',
	# 	'url':'https://www.eraserbenchmark.com/zipped/cose.tar.gz',
	# },
	# { #Check
	# 	'name':'evidence_inference',
	# 	'url':'https://www.eraserbenchmark.com/zipped/evidence_inference.tar.gz',
	# },
	# { #Check
	# 	'name': 'movies',
	# 	'url': 'https://www.eraserbenchmark.com/zipped/movies.tar.gz',
	# }
]

sample_objects_only=False
# sample_size=5000

sets = [('test', 'test.jsonl'),
		('train', 'train.jsonl'),
		('dev', 'val.jsonl'),
		]


def main():
	iprint(f'Downloading and processing eraser datasets to {eraser_dir}')
	object_sample = []
	for dataset in eraser_datasets:
		iprint(f'Processing {dataset["name"]}...')
		dataset_dir = os.path.join(eraser_dir, dataset['name'])
		extracted_dir = download_and_extract_eraser(dataset['url'], dataset_dir)
		doc_df = read_documents(extracted_dir)

		iprint('Document sample:')
		iprint(doc_df.head(5))

		for setnum, (setname, setfile) in enumerate(sets):
			iprint(f'{setname} set:')
			set_objects = read_jsonlines(os.path.join(extracted_dir, setfile))

			if sample_objects_only:
				sample_object = set_objects[0]
				sample_object['dataset'] = dataset
				sample_object['set'] = setname
				object_sample.append(sample_object)
				break

			set_objects = [process_set_object(set_object, doc_df, dataset["name"]) for set_object in set_objects]
			set_objects = [object for object in set_objects if object is not None]

			set_df = pd.DataFrame(set_objects, columns=['annotation_id', 'classification',
														'document', 'document_rationale_spans', 'document_rationale_values',
														'query', 'query_rationale_spans', 'query_rationale_values']).rename(columns={'annotation_id': 'id',
																																	 'classification': 'label'})
			iprint('Set sample:')
			iprint(set_df.head(5))

			# iprint('Evidences sample')
			# iprint(pformat(set_df.iloc[0]['evidences']))
			set_filepath = os.path.join(dataset_dir, f'{setname}.json')
			iprint(f'Writing to {set_filepath}')
			set_df.to_json(set_filepath, orient='records', lines=True)

			sample_filepath = os.path.join(dataset_dir, f'{setname}_sample.html')
			# sample_and_output_as_html(output_df=set_df,
			# 						  output_path=sample_filepath,
			# 						  sample_function=lambda df: df.sample(n=min(100,set_df.shape[0]), random_state=seed),
			# 						  text_span_value_columns={'document':{'document_rationale_spans':['document_rationale_values']},
			# 												   'query':{'query_rationale_spans':['query_rationale_values']}},
			# 						  scale_weird_ranges=False)

			iprint(f'Done with {setname} set')

		pass

	if sample_objects_only:
		write_jsonlines(object_sample, os.path.join(eraser_dir, 'object_sample.json'))

	iprint('Done!')


def process_set_object(set_object, doc_df, dataset_name):

	if len(set_object['evidences']) == 0: #Movies dataset has 1 or 2 empty examples
		return None

	# set_object['evidences'] is either a list of dictionaries nested in a 1-element list, or a list of 1-element lists with one evidence dict each.
	evidences = [evidence for sublist in set_object['evidences'] for evidence in sublist]



	if dataset_name == 'esnli':
		# e-snli is formatted differently from the others, where if the annotation id is '1007205537.jpg#1r1n', the premise and
		# hypothesis are always '1007205537.jpg#1r1n_premise' and '1007205537.jpg#1r1n_hypothesis', but these aren't listed as the
		# docid or query. So fill this in manually.

		document_id = set_object['annotation_id'] + '_premise'
		query_id = set_object['annotation_id'] + '_hypothesis'
		set_object['query'] = doc_df.loc[query_id]['document']

	else:
		if set_object.get('docids') is None:
			evidence_docids = list(set([evidence['docid'] for evidence in evidences]))
			assert len(evidence_docids) == 1
			document_id = evidence_docids[0]
		else:
			assert len(set_object['docids']) == 1
			document_id = set_object['docids'][0]

		query_id = None

	set_object['document'] = doc_df.loc[document_id]['document']


	doc_evidences = [evidence for evidence in evidences if evidence['docid'] == document_id]
	query_evidences = [evidence for evidence in evidences if evidence['docid'] == query_id]

	set_object['document_rationale_spans'], set_object['document_rationale_values'] = evidences_to_rationale(set_object['document'], doc_evidences)
	# assert set_object['document_rationale_spans'] is not None #We should never have a missing document rationale

	set_object['query_rationale_spans'], set_object['query_rationale_values'] = evidences_to_rationale(set_object['query'], query_evidences)

	# pprint(set_object)

	return set_object


def evidences_to_rationale(text, evidences, test=True):
	if evidences is None or len(evidences) == 0:
		return None, None

	tokens = text.split()

	spans = tokens_to_spans(tokens, text)

	values = [0.0 for token in tokens]
	for evidence in evidences:
		if test:
			spantext = text[spans[evidence['start_token']][0]:spans[evidence['end_token'] - 1][1]].replace('\n', ' ')
			if not spantext == evidence['text']:
				iprint(f'Mismatch: "{spantext}" vs "{evidence["text"]}"')
		for i in range(evidence['start_token'], evidence['end_token']):
			values[i] = 1.0

	# if len(evidences) > 1 and any([evidence['end_sentence'] > -1 for evidence in evidences]):
	# 	iprint(pformat(evidences))
	# 	x=1

	return spans, values


def read_documents(doc_dir):
	iprint(f'Reading documents from {doc_dir}')
	docfilename = [filename for filename in os.listdir(doc_dir) if filename in ['docs', 'docs.jsonl']][0]
	doc_objs = read_jsonlines_or_dir(os.path.join(doc_dir, docfilename))
	doc_df = pd.DataFrame(doc_objs)
	doc_df.set_index('docid', inplace=True)
	return doc_df


def download_and_extract_eraser(url, dataset_dir):
	iprint(f'Downloading and extracting {url} to {dataset_dir}')
	zip_filename = url.split('/')[-1]
	zip_filepath = os.path.join(dataset_dir, zip_filename)

	if os.path.exists(zip_filepath):
		iprint(f'Zip file already exists at {zip_filepath}')
	else:
		iprint(f'Downloading zip file from {url}...')
		subprocess.run(['wget', url, '-P', dataset_dir, '--no-check-certificate'])

	extracted_filename = zip_filename.split('.')[0]

	extracted_dir = look_for_extracted_dir(dataset_dir, extracted_filename)
	if extracted_dir:
		iprint(f'Extracted files already exist at {extracted_dir}')
	else:
		iprint('Extracting files')
		subprocess.run(['tar', '-xvf', zip_filepath, '-C', dataset_dir])
		extracted_dir = look_for_extracted_dir(dataset_dir, extracted_filename)
		if not extracted_dir:
			raise Exception('Cannot find extracted files after extracting them')

	return extracted_dir


def look_for_extracted_dir(dataset_dir, extracted_filename):
	if os.path.exists(os.path.join(dataset_dir, 'data', extracted_filename)):
		extracted_dir = os.path.join(dataset_dir, 'data', extracted_filename)
	elif os.path.exists(os.path.join(dataset_dir, extracted_filename)):
		extracted_dir = os.path.join(dataset_dir, extracted_filename)
	else:
		extracted_dir = None
	return extracted_dir


def read_jsonlines_or_dir(path):
	if os.path.isdir(path):
		return read_json_dir(path)
	else:
		return read_jsonlines(path)

def write_jsonlines(obj_list:List[Dict], filepath:str):
	iprint(f'Dumping list of {len(obj_list)} objects to {filepath}')
	with open(filepath,'w') as f:
		f.writelines(json.dumps(obj) for obj in obj_list)
	iprint('Done')

def read_jsonlines(filepath):
	iprint('Parsing jsonl file {}'.format(filepath))
	with open(filepath, 'r') as f:
		objs = [json.loads(line) for line in f.readlines()]
	iprint('{} items loaded.'.format(len(objs)), 1)
	return objs


doc_fn_patterns = [
	# re.compile("(?P<class>[a-z]+)R_(?P<docid>[0-9]+)\.txt"), #movie dataset
	# re.compile("(?P<docid>[A-Z]{2}_wiki_[0-9]+_[0-9]+)"), #boolq dataset
	# re.compile("(?P<docid>.+)\.txt"), #multirc dataset
	re.compile("(?P<docid>.+)")  # fever dataset and evidence_interference datasets
]


def read_json_dir(dirpath):
	iprint('Parsing doc dir {}'.format(dirpath))

	objs = []
	filenames = os.listdir(dirpath)
	for i, filename in enumerate(filenames):
		filepath = os.path.join(dirpath, filename)
		with open(filepath, 'r') as f:

			if filepath.endswith('.json'):
				obj = json.load(f)
			else:
				matched = False
				for pattern in doc_fn_patterns:
					m = re.match(pattern, filename)
					if m:
						matched = True
						obj = {'document': f.read()}
						obj.update(m.groupdict())
						objs.append(obj)
						if i == 0:
							iprint('{} --> {}'.format(filename, m.groupdict()), 1)
						break

				if not matched:
					raise Exception('Script does not know how to read file {}'.format(filepath))

	iprint('{} items loaded.'.format(len(objs)), 1)
	return objs


if __name__ == '__main__':
	main()

import subprocess
from util.misc_util import ensure_dir_exists, set_display_options
from util.print_util import iprint
import os
import pandas as pd
import nltk
import os
import json
from config.global_config import global_config

'''
Download and process wiki attack dataset and rationales

'''

#todo add citations


output_dir = f'{global_config["data_directory"]}/wiki_attack'

annotations_url = 'https://ndownloader.figshare.com/files/7554637'
comments_url = 'https://ndownloader.figshare.com/files/7554634'

dev_rationales_url = 'https://raw.githubusercontent.com/shcarton/rcnn/master/deliberativeness/data/processed/wiki/personal_attacks/wiki_attack_dev_rationale.csv'
test_rationales_url = 'https://raw.githubusercontent.com/shcarton/rcnn/master/deliberativeness/data/processed/wiki/personal_attacks/wiki_attack_test_rationale.csv'

tokenizer = nltk.tokenize.WordPunctTokenizer()


def main():
	set_display_options()
	ensure_dir_exists(output_dir)
	annotations_path = download(annotations_url, output_dir, 'annotations.tsv')
	comments_path = download(comments_url, output_dir, 'annotated_comments.tsv')
	dev_rationales_path = download(dev_rationales_url, output_dir)
	test_rationales_path = download(test_rationales_url, output_dir)

	comment_df = pd.read_csv(comments_path, delimiter='\t')
	annotation_df = pd.read_csv(annotations_path,delimiter='\t')

	mean_annotation_df = annotation_df.groupby(by='rev_id')['attack'].mean()

	combined_df = comment_df.join(mean_annotation_df, on='rev_id', how='inner')

	combined_df['label'] = combined_df['attack'].apply(lambda v: 'attack' if v>= 0.5 else 'nonattack')

	combined_df.rename(columns={"rev_id":"id",
								"comment":"document"},
					   inplace=True)

	sets = [('train', None),
		('dev', dev_rationales_path),
			('test',test_rationales_path)]

	for setname, rationale_path in sets:
		iprint(f'{setname} set')
		set_df = combined_df[combined_df['split'] == setname]
		set_df = set_df[['id','label','document']]
		iprint(f'{set_df.shape[0]} rows')
		set_df['document'] = set_df['document'].apply(replace_wiki_tokens)
		if rationale_path is not None:
			rationale_df = pd.read_csv(rationale_path)
			rationale_df['rationale'] = rationale_df['rationale'].apply(json.loads)
			iprint(f'{rationale_df.shape[0]} rationales found')
			rationale_df = rationale_df.merge(set_df, left_on='platform_comment_id', right_on='id', how='left')
			rationale_df['document_rationale_spans'] = rationale_df['document'].apply(lambda s:list(tokenizer.span_tokenize(s)))
			rationale_df.rename(columns={'rationale':'document_rationale_values'},inplace=True)
			assert (rationale_df['document_rationale_values'].apply(len) == rationale_df['document_rationale_spans'].apply(len)).all()

			set_df = set_df.merge(rationale_df[['id','document_rationale_values','document_rationale_spans']], on='id', how='left')

		rows = set_df.shape[0]
		set_df=set_df[set_df['document'] != '']
		iprint(f'{rows-set_df.shape[0]} empty texts removed, for final total of {set_df.shape[0]} rows')

		set_path = os.path.join(output_dir, setname+'.json')
		iprint(f'Writing to {set_path}')
		set_df.to_json(set_path, orient='records', lines=True)

		if setname in ['dev','test']:
			reduced_set_df = set_df[set_df['document_rationale_values'].notnull()].reset_index()
			reduced_set_path = os.path.join(output_dir, setname+'_with_rationales.json')
			iprint(f'Creating reduced {setname} set with ground truth rationales: {reduced_set_df.shape[0]} rows in reduced set.')
			iprint(f'Writing to {reduced_set_path}')
			reduced_set_df.to_json(reduced_set_path, orient='records', lines=True)


	iprint('Done!')



def replace_wiki_tokens(s):
	'''
	Replace tokens specific to the Wikipedia data with correct characters
	:param s:
	:return:
	'''
	return s.replace('TAB_TOKEN', '\t').replace('NEWLINE_TOKEN', '\n').strip()


def download(url, download_dir, filename:str=None):
	download_filename = url.split('/')[-1]
	download_path = os.path.join(download_dir, download_filename)

	if filename is not None:
		output_path = os.path.join(download_dir, filename)
	else:
		output_path = download_path

	iprint(f'Fetching file {output_path}')
	if not os.path.exists(output_path):
		if not os.path.exists(download_path):
			iprint(f'Downloading file from {url}...')
			subprocess.run(['wget', url, '-P' ,download_dir])
		else:
			iprint('File already downloaded')

		if filename is not None:
			iprint(f'Renaming to {output_path}')
			subprocess.run(['mv', download_path , output_path])
	else:
		iprint('File already exists')

	return output_path


if __name__ == '__main__':
	main()
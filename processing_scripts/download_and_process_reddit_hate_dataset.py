import os
from util.print_util import iprint
import subprocess
from util.misc_util import ensure_dir_exists, set_display_options
import json
import pandas as pd
import ast
import re
import numpy as np
import random

'''
Download and process the Reddit hate speech dataset available at https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech

Consists of 22k reddit comments
'''

url = 'https://raw.githubusercontent.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/master/data/reddit.csv'

output_dir = '../data/reddit_hate'

id_line_pattern = re.compile('(?P<indent>\t*)(?P<id>.+)\n')
text_line_pattern = re.compile('(?P<indent>\t*)(?P<text>.+)\n')
line_num_pattern = re.compile('^[0-9]+\. ', flags=re.MULTILINE)

seed = 9393
train_val_test_split = [.8,.1,.1]

def main():
	set_display_options()
	iprint('Downloading and processing Reddit hate speech dataset')
	ensure_dir_exists(output_dir)
	csv_path = download(url, output_dir)
	df = pd.read_csv(csv_path)

	# idx = 1
	# row = df.iloc[idx]
	# print('ID:')
	# print(row['id'])
	# print('Text:')
	# print(row['text'])
	# print('Hate speech indexes:')
	# print(row['hate_speech_idx'])
	# print('Responses:')
	# print(row['response'])

	comments = []
	comment_id =0
	for i, row in df.iterrows():
		id_lines = re.split(line_num_pattern, row['id'])
		id_lines= [id_line for id_line in id_lines if id_line != '']

		text_lines= re.split(line_num_pattern, row['text'])
		text_lines = [text_line for text_line in text_lines if text_line !='']

		assert len(id_lines) == len(text_lines), f'Error. {len(id_lines)} IDs found in: \n{row["id"]}\n...but {len(text_lines)} found in: \n{row["text"]}'

		if pd.isnull(row['hate_speech_idx']):
			hate_idxs = []
		else:
			hate_idxs = ast.literal_eval(row['hate_speech_idx'])


		#I'm not sure what the responses are, so I am not doing anything with them
		# if pd.isnull(row['response']):
		# 	responses = []
		# else:
		# 	responses = ast.literal_eval(row['response'])

		for j, (id_line, text_line) in enumerate(zip(id_lines, text_lines)):
			id_match = re.match(id_line_pattern, id_line)
			text_match  = re.match(text_line_pattern, text_line)
			x=0
			if j+1 in hate_idxs:
				label= 'hate'
			else:
				label = 'nonhate'

			indent = len(id_match['indent'])
			if indent == 0.5:
				x=0
			comment = {'id':comment_id,
					   'thread_id':i,
					   'id_in_thread':j,
					   'reddit_id':id_match['id'],
					   'depth':indent,
					   'label':label,
					   'document':text_match['text'].replace('  ','\n')}
			comments.append(comment)
			comment_id += 1

	comment_df = pd.DataFrame(comments)

	thread_gb = comment_df.groupby('thread_id')
	thread_ids = list(thread_gb.indices.keys())

	thread_id_sets = create_splits(thread_ids, train_val_test_split, seed)
	sets = list(zip(['train','val','test'], thread_id_sets))
	sets.append(('all',thread_ids))

	for setname, set_thread_ids in sets:
		set_comment_ids = [comment_id for thread_id in set_thread_ids for comment_id in thread_gb.indices[thread_id]]
		set_df = comment_df.iloc[set_comment_ids]
		iprint(f'{setname} set: {set_df.shape[0]} comments from {len(set_thread_ids)} threads: {100*set_df.shape[0]/comment_df.shape[0]:.3f}% of whole')
		set_filename = f'{setname}.json'
		set_path = os.path.join(output_dir, set_filename)
		iprint(f'Outputting to {set_path}')
		set_df.to_json(set_path, orient='records', lines=True)

	iprint('Done!')


def create_splits(sequence, fractions, seed):
	iprint(f'Creating splits for sequence of length {len(sequence)} with following fractions: {fractions}')
	np.random.seed(seed)
	shuffled_sequence = list(sequence)
	np.random.shuffle(shuffled_sequence)
	assert np.sum(fractions) == 1

	subsequences = []
	cumulative_fraction_sum = 0
	for fraction in fractions:
		start_fraction = cumulative_fraction_sum
		end_fraction = cumulative_fraction_sum+fraction

		start_index = int(start_fraction * len(shuffled_sequence))
		end_index = int(end_fraction * len(shuffled_sequence))
		subsequence = shuffled_sequence[start_index : end_index]
		iprint(f'Split of length {len(subsequence)} with start and end: {start_index}, {end_index}')

		subsequences.append(subsequence)
		cumulative_fraction_sum += fraction

	return subsequences



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
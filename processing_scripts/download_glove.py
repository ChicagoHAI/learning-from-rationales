from util.print_util import iprint
from util.misc_util import ensure_containing_dir
import os
import subprocess
from config.global_config import global_config

glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
file_names = ["glove.6B.100d.txt", "glove.6B.200d.txt", "glove.6B.300d.txt", "glove.6B.50d.txt"]
dest_dir = f'{global_config["pretrained_directory"]}/glove'

'''
Download and process GLOVE word vectors. Only needed for LSTM model variants. 
'''

def main():
	iprint(f'Downloading GloVe vectors from {glove_url} to {dest_dir}')
	ensure_containing_dir(dest_dir)

	zip_filename = glove_url.split('/')[-1]
	zip_filepath = os.path.join(dest_dir, zip_filename)

	if os.path.exists(zip_filepath):
		iprint(f'Zip file already exists at {zip_filepath}')
	else:
		iprint(f'Downloading zip file from {glove_url}...')
		subprocess.run(['wget', glove_url, '-P', dest_dir])

	if all([os.path.exists(os.path.join(dest_dir, file_name)) for file_name in file_names]):
		iprint('Zip file already extracted')
	else:
		iprint('Unzipping zip file')
		subprocess.run(['unzip', zip_filepath, '-d', dest_dir])

	iprint('Done!')


if __name__ == '__main__':
	main()

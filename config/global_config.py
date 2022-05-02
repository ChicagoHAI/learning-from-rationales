import os
'''
global config options
'''

project_root =os.path.dirname(os.path.dirname(__file__))
global_config = {
	'data_directory':os.path.abspath(f'{project_root}/data'),
	'output_directory':os.path.abspath(f'{project_root}/output'),
	'huggingface_cache':os.path.abspath(f'{project_root}/huggingface_cache'),
	'pretrained_directory':os.path.abspath(f'{project_root}/pretrained'),

}
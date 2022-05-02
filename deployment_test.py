import os
import torch
import pytorch_lightning as pl
import transformers
# import torch_scatter
import pytz
import pandas as pd
import nltk

from config.global_config import global_config
from pprint import pprint


print('Looks like all imports worked...')
print('CWD/project root: ',os.getcwd())
print('CUDA available: ', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)

print('Global config options (must be hardcoded to change):')
pprint(global_config)

print('Done!')
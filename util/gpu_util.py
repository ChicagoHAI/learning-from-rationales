import gc
import subprocess
import re
import os
from pprint import pprint, pformat
import xml.etree.ElementTree as et
# import torch

# all_gpus = [0, 1,2,3]

# A line will look something like this: |    3      3395    C   python                                         547MiB |
# gpu_list_pattern = 'GPU (?P<gpu>[0-9]+):.+'
# import torch
# from pytorch_lightning.utilities.memory import garbage_collection_cuda

smi_gpu_line1_pattern = '\| +(?P<gpu_num>[0-9]+) +(?P<gpu_name>[0-9A-Za-z\. ]+) +[A-Za-z]+ +\|' #I don't really care about anything after the GPU name
smi_gpu_line2_pattern = '\| +(?P<fan_perc>[0-9N/A%]+) +(?P<temp>[0-9]+)C +(?P<perf>[A-Z0-9]+) +(?P<pwr_usage>[0-9]+)W / (?P<pwr_cap>[0-9WN/A]+) +\| +(?P<mem_usage>[0-9]+)MiB / (?P<mem_cap>[0-9]+)MiB.+' #Don't care past memory usage
smi_process_pattern = '\| +(?P<gpu_num>[0-9]+) +[a-zA-Z0-9/]+ +[a-zA-Z0-9/]+ +(?P<pid>[0-9]+) +[CG] +\S+ +(?P<mem_usage>[0-9]+)[a-zA-Z]+ +\|'

def parse_nvidia_smi_output(smi_str:str):
	'''
	Parse the output of the nvidia-smi command line command, because for whatever reason pytorch is garbage at
	figuring out GPU memory usage & similar
	:param smi_str:
	:return:
	'''

	lines = smi_str.split('\n')

	divider_line_num = [line_num for line_num, line in enumerate(lines) if all([char == ' ' for char in line])][0]
	sec1 = '\n'.join(lines[:divider_line_num])
	sec2 = '\n'.join(lines[divider_line_num:])

	result = {'gpus':[], 'processes':[]}

	line1s = list(re.finditer(smi_gpu_line1_pattern, sec1))
	line2s = list(re.finditer(smi_gpu_line2_pattern, sec1))
	process_lines = list(re.finditer(smi_process_pattern, sec2))

	# print('Section 1:')
	# print(sec1)
	# print('Section 2:')
	# print(sec2)
	# print('=======================================================')

	assert len(line1s) == len(line2s)
	for info1, info2 in zip(line1s,line2s):
		result['gpus'].append({'gpu_num':int(info1['gpu_num']),
							   'gpu_name':info1['gpu_name'].strip(),
							   'mem_usage':int(info2['mem_usage']),
							   'mem_cap':int(info2['mem_cap'])
							   })
	for process in process_lines:
		result['processes'].append({'gpu_num':int(process['gpu_num']),
									'pid':int(process['pid']),
									'mem_usage':int(process['mem_usage'])
									})

	return result

def get_old_gpu_usage(verbose=True):
	smi_str = subprocess.check_output(['nvidia-smi']).decode("utf-8")
	# gpus_in_use = []
	if verbose:
		print('Output of nvidia-smi:')
		print(smi_str)
		print('GPUs process ownership details:')

	try:
		smi_dict = parse_nvidia_smi_output(smi_str)
		return smi_dict
	except:
		return None


def get_gpu_usage(verbose=True):
	smi_str = subprocess.check_output(['nvidia-smi']).decode("utf-8")
	xml_str = subprocess.check_output(['nvidia-smi', '-q', '-x']).decode("utf-8")

	# gpus_in_use = []
	if verbose:
		print('Output of nvidia-smi:')
		print(smi_str)
		print('GPUs process ownership details:')

	smi_dict = parse_nvidia_smi_xml_output(xml_str)
	return smi_dict


'''
{'gpu_num':int(info1['gpu_num']),
							   'gpu_name':info1['gpu_name'].strip(),
							   'mem_usage':int(info2['mem_usage']),
							   'mem_cap':int(info2['mem_cap'])
							   }
'''
def parse_nvidia_smi_xml_output(xml_str):
	tree = et.fromstring(xml_str)

	gpus = tree.findall('gpu')
	gpu_dict_list = []
	for gpu in gpus:
		gpu_dict = {'gpu_num':int(gpu.find('minor_number').text),
					'gpu_name':gpu.find('product_name').text,
					'mem_usage':int(gpu.find('fb_memory_usage').find('used').text.split(' ')[0]),
					'mem_cap':int(gpu.find('fb_memory_usage').find('total').text.split(' ')[0]),
					}
		gpu_dict_list.append(gpu_dict)

	return {'gpus':gpu_dict_list, 'processes':[]}

	pass

def analyze_gpu_usage(verbose=True):
	'''
	Runs the nvidia-smi system command and parses the output as a list of GPU numbers that are being used. Also
	looks up each process that is using a GPU and returns some information about it.
	:param verbose:
	:return:
	'''
	smi_dict = get_gpu_usage(verbose=verbose)


	for process in smi_dict['processes']:
		gpu = process['gpu_num']
		mbs = process['mem_usage']
		pid = process['pid']
		ps_str = subprocess.check_output('ps -u -p {}'.format(pid),shell=True).decode("utf-8")
		header, info,_ = ps_str.split('\n')
		if verbose:
			print('\tGPU\t{}'.format(header))
			print('\t{}\t{}'.format(gpu, info))
		# except Exception as ex:
		# 	pass


	return smi_dict

# default_gpus=None, force_default=False, number_of_gpus_wanted=1,
def choose_gpus(verbose=True, num= 1, max_utilization = 18000, align_with_cuda_visible=False):
	'''
	Check which GPUs are in use and choose one or more that are not in use
	:param verbose: whether to print out info about which GPUs are being used
	:param default_gpus: list of default number(s) to go with if it is available
	:param force_default: whether to force the use of the default GPU(s)
	:param number_of_gpus_wanted: whether to force the use of the default GPU(s)

	:return:
	'''
	# smi_str = subprocess.check_output(['nvidia-smi'])
	# gpu_process_tuples = re.findall(pattern, smi_str)

	print('Choosing one or more GPUs to use')
	if num is None:
		num=1

	try:
		smi_dict = analyze_gpu_usage(verbose=verbose)
		assert smi_dict is not None
	except Exception as ex:
		print(f'Exception caught while trying parse GPU usage: {ex}\nReturning GPU 0 by default.')
		return [0]


	gpus_by_use = sorted([gpu for gpu in smi_dict['gpus']], key=lambda g: g['mem_usage'])
	# chosen_gpus = gpus_by_use[0:num]
	gpu_nums = [gpu['gpu_num'] for gpu in gpus_by_use if gpu['mem_usage'] <= max_utilization]
	if verbose:print(f'Available GPUs with less than {max_utilization}M: {gpu_nums}')

	if align_with_cuda_visible:
		if 'CUDA_VISIBLE_DEVICES' in os.environ:
			visible_gpu_nums = [int(numstr) for numstr in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
			if verbose:print(f'CUDA visible GPUs: {visible_gpu_nums}. Aligning available GPUs with this list')
			gpu_nums = [visible_gpu_nums.index(num) for num in gpu_nums if num in visible_gpu_nums]
		else:
			if verbose:print('No CUDA_VISIBLE_DEVICES environment variable found.')

	chosen_gpu_nums = gpu_nums[0:num]

	if len(chosen_gpu_nums) == 0:
		print(f'Could not find a GPU with less than {max_utilization} MBs utilization')
		raise Exception
	elif len(chosen_gpu_nums) < num:
		if verbose: print(f'Could only find {len(chosen_gpu_nums)} gpus with less than {max_utilization} MBs utilization: {chosen_gpu_nums}')
		return chosen_gpu_nums
	else:
		if verbose: print(f'Chose GPUS {chosen_gpu_nums}')
		return chosen_gpu_nums



	# if default_gpus is not None and all([default_gpu not in gpus_in_use for default_gpu in default_gpus]):
	# 	if verbose: print('Default GPUs {} are available, so selecting them'.format(default_gpus))
	# 	return default_gpus
	# else:
	# 	if len(all_gpus) - len(gpus_in_use) >= number_of_gpus_wanted:
	# 		chosen_gpus = [gpu for gpu in all_gpus if gpu not in gpus_in_use][:number_of_gpus_wanted]
	# 		if verbose: print('Selected GPUS {} for use'.format(chosen_gpus))
	# 		return chosen_gpus
	# 	else:
	# 		raise Exception('Could not find {} free GPUs. Check output of nvidia-smi command for details.'.format(number_of_gpus_wanted))


def choose_and_set_available_gpus(verbose=True, manual_choice=None, max_utilization=5000):
	'''
	Choose some GPUs and make only them visible to CUDA
	:param verbose:
	:param default_gpu:
	:param force_default:
	:param number_of_gpus_wanted:
	:return:
	'''

	if manual_choice is None:
		chosen_gpus = choose_gpus(verbose,max_utilization=max_utilization)
	else:
		chosen_gpus = manual_choice

	print('Setting GPU(s) {} as only visible device.'.format(chosen_gpus))
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in chosen_gpus])

# import tensorflow as tf
# def generate_tensorflow_config(lowmem = True):
# 	config = tf.ConfigProto()
# 	config.gpu_options.allow_growth = lowmem
# 	return config

def main():
	print('Checking GPU usage and choosing one')
	# gpus = analyze_gpu_usage()
	gpus = choose_gpus(verbose=True, num=1, max_utilization=23000, align_with_cuda_visible=True)



	print(gpus)

if __name__ == '__main__':
	main()
existing_hashes = set()


def inspect_gpu_memory(track_hashes=True, inspect_gc=False, inspect_pytorch=False, inspect_nvidia=True):
	print('\nInspecting GPU memory.')

	if inspect_gc:
		numobjs =0
		numsizes = 0
		nbytes = 0
		grads= 0
		gradbytes = 0
		num_matches =0
		num_mismatches =0
		global existing_hashes
		hash_size =len(existing_hashes)
		for obj in gc.get_objects():
			try:
				if torch.is_tensor(obj):
					tensor = obj
				elif (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
					tensor = obj.data
				else:
					tensor = None

				if tensor is not None:
					if hasattr(tensor,'device') and tensor.device is not None and hasattr(tensor.device, 'type')  and tensor.device.type == 'cuda':
						numobjs +=1
						if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
							numsizes += 1
							nbytes += tensor.numel() * tensor.element_size()

						if hasattr(tensor, '__hash__') and track_hashes:

							if tensor.__hash__() not in existing_hashes:
								num_mismatches += 1
								existing_hashes.add(tensor.__hash__())
								# if hash_size > 0:
								# 	print(f'Mismatch found of shape {tensor.shape}')
								# 	pass
							else:
								num_matches +=1

						if hasattr(tensor, 'grad'):
							grads += 1
							if hasattr(tensor.grad, 'numel') and hasattr(tensor.grad, 'element_size'):
								gradbytes += tensor.grad.numel() * tensor.grad.element_size()


			except: pass


		print(f'{numobjs} objects on GPU, {numsizes} have discernable sizes, comprising {nbytes} bytes ({nbytes//2**20}M)')
		print(f'{grads} grad objects found comprising comprising {gradbytes} bytes ({gradbytes//2**20}M)')
		if track_hashes: print(f'{num_mismatches} misses on existing tensor hash set of size ({hash_size}), {num_matches} hits.')

	if inspect_gc:
		pytorch_memory_allocation = {gpu:torch.cuda.memory_allocated(gpu)//2**20 for gpu in range(torch.cuda.device_count())}
		print(f'Pytorch memory allocated (M): {pytorch_memory_allocation}')

	if inspect_nvidia:
		usage_dict = get_gpu_usage(verbose=False)
		print('nvidia-smi GPU usage:',pformat({gpu['gpu_num']:gpu['mem_usage'] for gpu in usage_dict['gpus']}))

	return
import inspect
import logging
import logging.handlers
import sys
from datetime import datetime
import pytz
import os
from typing import Union, List

default_logger_name = 'logging'
logging_initialized = False
default_formatter = logging.Formatter("%(asctime)s - %(message)s",
									  datefmt='%m/%d/%Y %I:%M %p')

def initialize_logging(directories:Union[str, List[str]]=None, filename:str= 'log.txt', logger_name:str= ''):
	'''
	Initialize the logger. Have it print to stdout and optionally a file
	:param filepath:
	:param logger_name:
	:return:
	'''
	global logging_initialized

	global default_logger_name
	default_logger_name = logger_name
	logger = logging.getLogger(logger_name)
	if not logging_initialized:

		#remove existing handlers so we don't end up with extra stdout streams and stuff
		for handler in list(logger.handlers):
			logger.removeHandler(handler)

		logger.setLevel(logging.INFO)

		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(default_formatter)
		console_handler.setLevel(logging.INFO)
		logger.addHandler(console_handler)
		# err_handler = logging.StreamHandler(sys.stderr)
		# err_handler.setFormatter(default_formatter)
		# logger.addHandler(err_handler)

		logging_initialized = True

	if directories is not None:
		if type(directories) != list:
			directories = [directories]

		for directory in directories:
			if directory is not None:
				filepath = os.path.join(directory, filename)
				add_log_destination(filepath=filepath, logger=logger, formatter=default_formatter)

	iprint(f'Logging initialized. Logging to console and following directories: {directories}')
	return logger

def retrieve_file_handlers(logger:logging.Logger, filepath:str):
	rval = []
	filepath = os.path.abspath(filepath)
	for handler in logger.handlers:
		if hasattr(handler, 'baseFilename') and handler.baseFilename == filepath:
			rval.append(handler)
	return rval

def add_log_destination(directory:str=None,
						filepath:str=None,
						filename:str='log.txt',
						logger:logging.Logger=None,
						formatter:logging.Formatter=default_formatter,
						logger_name:str=''):


	if logger is None:
		logger = logging.getLogger(logger_name)

	if filepath is None:
		filepath = os.path.join(directory, filename)

	existing_handlers = retrieve_file_handlers(logger, filepath)
	if len(existing_handlers) > 0:
		iprint('One or more handlers already exist for this file. Not adding another one.')
		return

	file_handler = logging.handlers.RotatingFileHandler(
		filepath, maxBytes=(1048576*5), backupCount=7
	)
	file_handler.setFormatter(formatter)
	file_handler.setLevel(logging.INFO)
	logger.addHandler(file_handler)

def remove_log_destination(directory:str,
						   filename:str='log.txt',
						   logger:logging.Logger=None,
						   logger_name:str=''):
	if logger is None:
		logger = logging.getLogger(logger_name)

	filepath = os.path.abspath(os.path.join(directory, filename))

	handlers = retrieve_file_handlers(logger, filepath)
	removed=0
	for handler in handlers:
		logger.removeHandler(handler)
		removed += 1

	# if removed ==0:
	# 	iprint('No handlers removed...')

# def remove_extraneous_stream_handlers(logger:logging.Logger):
# 	stream_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout]
# 	if len(stream_handlers) > 1:
# 		for i in range(1,len(stream_handlers)):
# 			logger.removeHandler(stream_handlers[i])
# class StreamToLogger(object):
# 	"""
# 	Fake file-like stream object that redirects writes to a logger instance.
# 	Taken from https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
# 	"""
# 	def __init__(self, logger, level):
# 		self.logger = logger
# 		self.level = level
# 		self.linebuf = ''
#
# 	def write(self, buf):
# 		for line in buf.rstrip().splitlines():
# 			self.logger.log(self.level, line.rstrip())
#
# 	def flush(self):
# 		pass


def do_nothing():
	pass



do_print = True #Global variable that gets manipulated when we want to silence output
iprint_ignore_frames=3

def iformat(o,indent_token = '| ', inc=0, add_timestamp=False):
	num_indent = calculate_indentation(ignore=iprint_ignore_frames) + inc
	indent = indent_token*num_indent

	o = str(o)
	if '\n' in o:
		o = '\n'+o
	o = indent + o.replace('\n', '\n' + indent)

	if add_timestamp:
		o = fdatetime(now())+': ' + o

	return o

def iprint(o,
		   dynamic_indent=True,
		   indent_token = '| ',
		   inc=0,
		   add_timestamp=False,
		   log_level=logging.INFO):
	'''
	Print with indendation matching how deeply nested the call is in either local control flow or the call stack

	Basically just makes it a little more visually evident where printed output is coming from
	:param o:
	:param dynamic_indent:
	:param indent_token:
	:param inc:
	:return:
	'''

	if do_print:
		if dynamic_indent:
			o=iformat(o, indent_token=indent_token, inc=inc, add_timestamp=add_timestamp)


		if logging_initialized:
			logger = logging.getLogger(default_logger_name)
			# if check_for_multiple_stream_handlers:
			# 	remove_extraneous_stream_handlers(logger)
			logger.log(log_level, o)
		else:
			print(o)
	pass

def iiprint(*args, **kwargs):
	iprint(*args, log_level=logging.INFO, **kwargs)


def calculate_indentation(ignore=1, verbose=False):
	'''
	Calculate how far we want to indent a print statement based on how deep it is in the stack
	and how deeply indented it and its parents are (to account for loops and conditionals)

	:param ignore:
	:return:
	'''
	stack = inspect.stack()
	i  = ignore #ignore this many frames of the stack
	indent = 0
	while True:

		if i >= len(stack):
			break


		numtabs = stack[i].code_context[0].count('\t') - 1 if stack[i].code_context is not None else 0
		numtabs = max(0,numtabs)

		indent += numtabs

		if (stack[i].function == 'main'):
			break

		i+=1
		indent += 1

	return max(0,indent)

def set_print(state):
	global do_print
	previous = do_print
	do_print = state
	return previous

# ect = pytz.timezone('US/Eastern')
ect = pytz.timezone('US/Mountain')

def now():
	now = datetime.now(ect)
	# formatted = now.strftime("%Y-%m-%d %I:%M %p %z")
	# now.strftime = lambda self, format:formatted
	return now

def today():
	now = datetime.now(ect)
	today = now.date()
	return today


ticks = {}


def tick(key='', verbose=True):
	current_time = now()

	if verbose: iprint(f'Tick. {key}')

	ticks[key] = current_time


def tock(key='', comment=None, verbose=True):
	'''
	Convenience function for printing a timestamp with a comment
	:param comment:

	:return:
	'''
	last_tick = ticks[key]

	current_tick = now()


	ps = f'Tock. {finterval(current_tick-last_tick)} elapsed. {key}'

	if verbose: iprint(ps)

	return current_tick-last_tick

def ftime(dt):
	return dt.strftime("%I:%M %p")

def fdatetime(dt):
	return dt.strftime("%I:%M %p %m/%d/%Y")


def finterval(interval):
	return str(interval)



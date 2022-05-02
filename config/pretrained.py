from config.global_config import global_config

pretrained = {
	'glove_100':{'path':f'{global_config["pretrained_directory"]}/glove/glove.6B.100d.txt'},
	'glove_200':{'path':f'{global_config["pretrained_directory"]}/glove/glove.6B.200d.txt'},
	'glove_300':{'path':f'{global_config["pretrained_directory"]}/glove/glove.6B.300d.txt'},
	'glove_50':{'path':f'{global_config["pretrained_directory"]}/glove/glove.6B.50d.txt'}
}
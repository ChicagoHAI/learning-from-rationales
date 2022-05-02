from model.bert_classification_model import BertClassificationModel
# from model.logistic_regression_model import LogisticRegressionModel
# from model.lstm_classification_model import LSTMClassificationModel
from model.bert_rationale_model import BertRationaleModel

def resolve_model_class(classname:str):
	if classname == 'BertClassificationModel':
		return BertClassificationModel
	# elif classname == 'LogisticRegressionModel':
	# 	return LogisticRegressionModel
	# elif classname == 'LSTMClassificationModel':
	# 	return LSTMClassificationModel
	elif classname == 'BertRationaleModel':
		return BertRationaleModel
	else:
		raise Exception(f'Unknown model class "{classname}"')
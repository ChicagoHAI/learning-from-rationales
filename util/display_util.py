from typing import Callable, List, Sequence, Dict, Union
import pandas as pd
import numpy as np
from util.print_util import iprint

from model_component.binarize import is_binary_rationale

#borrowed from https://www.schemecolor.com/blue-red-yellow-green.php
blue = "53, 129, 216"
darkblue = "42, 99, 164"
green="99, 202, 216"
red= "216, 46, 63"
yellow="255, 225, 53"
darkyellow="191, 171, 40"


def sample_and_output_as_html(
		output_df: pd.DataFrame,
		output_path: str,
		sample_function: Callable = None, #function defining how to sample output d
		tokens_rationale_columns: Dict[str, List[str]] = None, # mapping of token column(s) to rationale column(s) to render
		text_span_value_columns: Dict[str, Dict[str, List[str]]] = None, #mapping of text column(s) to span column(s) to rationale value column(s) to render
		rationalized_prediction_sets: List[Dict[str,str]]=None, #list of sets of prediction, rationale, tokens, and true class to render
		drop: bool = True,
		color_predictions:bool=False,
):


	if sample_function is None:
		sample_df = output_df
	else:
		sample_df = sample_function(output_df)

	iprint(f'Sampling {sample_df.shape} rows out of {output_df.shape} to HTML at {output_path}')
	drop_columns = set()


	if tokens_rationale_columns is not None:
		for tokens_column in tokens_rationale_columns:
			iprint(f'Tokens column: {tokens_column}')

			rationale_columns = tokens_rationale_columns[tokens_column]
			for rationale_column in rationale_columns:
				iprint(f'Rationale column: {rationale_column}')
				sample_df[f'{rationale_column}_text'] = sample_df[[tokens_column, rationale_column]].apply(
					lambda s: generate_rationalized_text(tokens=s[tokens_column],
														 rationale=s[rationale_column],
														 # rationale_weight=s['rationale_weight'],
														 scale_rationale= not is_binary_rationale(rationale_column)), axis=1)
				if drop: drop_columns.add(rationale_column)
			if drop: drop_columns.add(tokens_column)

	#%%
	if text_span_value_columns is not None:
		for text_column in text_span_value_columns:
			iprint(f'Text column: {text_column}')
			for span_column in text_span_value_columns[text_column]:
				iprint(f'Span column: {span_column}')
				for value_column in text_span_value_columns[text_column][span_column]:
					iprint(f'Value column: {value_column}')
					sample_df[f'{text_column}_{value_column}_text'] = sample_df[[text_column, span_column, value_column]].apply(
						lambda s: generate_rationalized_text(text=s[text_column],
															 spans=s[span_column],
															 rationale=s[value_column],
															 # rationale_weight=s['rationale_weight'],
															 scale_rationale= not is_binary_rationale(rationale_column)), axis=1)
					if drop: drop_columns.add(value_column)
				if drop: drop_columns.add(span_column)
			if drop: drop_columns.add(text_column)
	#%%

	if rationalized_prediction_sets is not None:
		for rationalized_prediction_set in rationalized_prediction_sets:
			tokens_column = rationalized_prediction_set['tokens']
			rationale_column = rationalized_prediction_set['rationale']
			py_index_column = rationalized_prediction_set['py_index']
			y_index_column = rationalized_prediction_set['y_index']
			sample_df[f'{rationale_column}_text'] = sample_df.apply(
				lambda s: generate_rationalized_text(tokens=s[tokens_column],
													 rationale=s[rationale_column],
													 py_index=s.get(py_index_column),
													 y_index=s.get(y_index_column),
													 # rationale_weight=s['rationale_weight'],
													 scale_rationale= not is_binary_rationale(rationale_column)), axis=1)

			drop_columns.add(tokens_column)
			drop_columns.add(rationale_column)


	if color_predictions:
		for column in sample_df.columns:
			if column.endswith('py_index'):
				sample_df[column] = sample_df.apply(lambda s:color_prediction_column(py_index=s[column], y_index=s['label']), axis=1)

	if drop:
		for drop_column in drop_columns:
			if drop_column in sample_df.columns:
				sample_df.drop(columns=drop_column, inplace=True)

	df_to_html(sample_df, output_path)


def color_prediction_column(py_index:float, y_index:float):
	if y_index is not None and py_index is not None:
		if y_index == py_index:
			div_style=f'style="background-color: rgba({green},0.5);"'
		else:
			div_style = f'style="background-color: rgba({red},0.5);"'
	else:
		div_style=''

	return f'<div {div_style}> {py_index} </div>'



def df_to_html(html_df: pd.DataFrame, output_path: str):
	output_str = '''<html>
    {}
    <body><table>'''.format(rationale_highlight_style())

	# table headers
	output_str += '\n<tr>'
	output_str += '\n<th class="val_column">  </th>'
	for column_name in html_df.columns:
		th_class = 'text_column' if column_name.endswith('_text')  or  column_name.endswith('_html') else 'val_column'
		output_str += f'\n<th class="{th_class}"> {column_name} </th>'
	output_str += '\n</tr>'

	# table rows
	for row_name, row in html_df.iterrows():
		output_str += '\n<tr>'
		output_str += f'\n<td class="val_cell"><div class="val_contents"> {row_name}</div> </td>'
		for column_name in html_df.columns:
			td_class = 'text_cell' if column_name.endswith('_text') or column_name.endswith('_html') else 'val_cell'
			div_class = 'text_contents' if column_name.endswith('_text') or column_name.endswith('_html') else 'val_contents'
			if type(row[column_name]) == float:
				output_str += f'\n<td class="{td_class}"> <div class="{div_class}">{row[column_name]:.3f}</div> </td>'
			else:
				output_str += f'\n<td class="{td_class}"> <div class="{div_class}"> {row[column_name]} </div> </td>'

		output_str += '\n</tr>'

	output_str += '\n</table></body></html>'

	with open(output_path, 'w') as f:
		f.write(output_str)


def generate_rationalized_text(rationale: Sequence,
							   text: str = None,
							   spans: Sequence[List[int]] = None,
							   tokens: Sequence[str] = None,
							   y_index:int=None,
							   py_index:int=None,
							   scale_rationale:bool=False,
							   fill_whitespaces:bool=False):
	'''
	Convert one set of tokens and one rationale to an HTML string of rationalized text
	:param tokens:
	:param rationale:
	:return:
	'''

	if tokens is None:
		if spans is not None:
			tokens = [text[span[0]:span[1]] for span in spans]
		else:
			return text

	tokens = [token.replace('Ġ', '') for token in tokens]  # for cases where the underlying tokenizer is a RobertaTokenizer

	rationale = [val if val is not None else 0.0 for val in rationale]
	minr = min(rationale)
	maxr = max(rationale)
	ranger = maxr - minr

	# if scale_weird_ranges and (ranger > 2 or (ranger < 0.5 and ranger > 0.01)):
	# 	scale_func = lambda zi: (zi - minr) / max(ranger, 0.001)
	# 	scaled = True
	# else:
	# 	scale_func = lambda zi: zi
	# 	scaled = False

	if scale_rationale:
		scale_func = lambda zi: (zi - minr) / max(ranger, 0.001)
		scaled = True
	else:
		scale_func = lambda zi: zi
		scaled = False

	#color the containing div green or red based on whether we can identify if it was a correct or incorrect prediction
	if y_index is not None and py_index is not None:
		if y_index == py_index:
			div_style=f'style="background-color: rgba({green},0.5);"'
		else:
			div_style = f'style="background-color: rgba({red},0.5);"'
	else:
		div_style=''

	# z = series.iloc[1]
	if rationale is not None and not np.any(pd.isnull(rationale)):
		assert (len(tokens) == len(rationale))

		comment_html = f'<div {div_style}>'

		def rgba_func(zi):
			scaled_zi = scale_func(zi)
			if scaled_zi > 0:
				rgba = f'rgba({darkblue if scaled else blue},{scaled_zi:.3f})'
			else:
				rgba = f'rgba({darkyellow if scaled else yellow},{abs(scaled_zi):.3f})'

			return rgba

		for i, (token, zi) in enumerate(zip(tokens, rationale)):
			# Red for positive, green for negative

			comment_html += f'<span class="rationale tooltip" style="background-color: {rgba_func(zi)};">'
			# comment_html += text[token_start:token_end]
			comment_html += token
			comment_html += '<span class="tooltiptext">{:.3f}</span>'.format(zi)
			comment_html += '</span>'

			whitespace = ' '
			if spans is not None and len(spans) > i+1:
				# if spans[i+1][0] != spans[i][1]:
				whitespace = text[spans[i][1]:spans[i+1][0]]

			if fill_whitespaces:
				whitespace_zi = min(zi, rationale[i+1] if i < len(spans)-1 else 0.0)
				whitespace = f'<span class="rationale whitespace" style="background-color: {rgba_func(whitespace_zi)};">' + whitespace + '</span>'
				# if zi > 0.5 and rationale[i+1] > 0.5 and i < len(spans):
				# if tokens[i] == 'hurt' and i < len(tokens)-1 and tokens[i+1] == 'anyone':
				# 	x=0

			comment_html += whitespace


			# if len(tokens) > i+1 and tokens[i+1] == '.':
			# 	x=0

		comment_html += '</div>'
	else:
		comment_html = '<div>{}</div>'.format(' '.join(tokens))

	return comment_html


def rationale_highlight_style():
	output_str = '''
        <style>
        .rationale {
        background-color:LightCoral;
        }
        table, tr, th, td{
        border:1px solid black;
          border-collapse: collapse;

        }

        /* Tooltip container */
        .tooltip {
          position: relative;
          display: inline-block;
          border-bottom: 1px dotted black; /* If you want dots under the hoverable text */
        }

        /* Tooltip text */
        .tooltip .tooltiptext {
          visibility: hidden;
          background-color: black;
          color: #fff;
          text-align: center;
          padding: 5px;
          border-radius: 6px;

          /* Position the tooltip text - see examples below! */
          position: absolute;
          z-index: 1;
        }

        /* Show the tooltip text when you mouse over the tooltip container */
        .tooltip:hover .tooltiptext {
          visibility: visible;
        }
        		.val_contents {
		max-width:100px;
		max-height:400px;
		overflow:auto;
		}
		.text_contents{
		min-width:400px;
		}
		
		/* Have the column headers float at the top of the page*/
		.val_column, .text_column{
		position: sticky;
		background:rgba(200, 200, 200, 0.9);
		top:0;
		z-index: 1;
	
        </style>'''

	return output_str


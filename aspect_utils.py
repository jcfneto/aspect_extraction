import json
import evaluate

import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict

from transformers import (AutoTokenizer,
						  DataCollatorForTokenClassification,
						  TFAutoModelForTokenClassification,
						  TFBertForTokenClassification)


def json_to_dataframe(path: str) -> pd.DataFrame:
    """Transforms json data file into dataframe.

    Args:
        path: Path of json file.

    Retuns:
        df: Dataframe with data.
    """
    wrk = pd.read_json(path)
    return pd.json_normalize(wrk.reviews)


def split(text: str) -> list:
    """Separates string from list of string into list of tokens.

    Args:
        text: String with the sentence to be split.

    Returns:
		List with tokens.
    """
    return text.strip('][').split(', ')


def build_tokenizer(model_checkpoint: str):
    """Initializes the tokenizer.

    Args:
        model_checkpoint: Nome do modelo.

     Returns
    """
    return AutoTokenizer.from_pretrained(model_checkpoint)


def align_labels_with_tokens(labels: list, word_ids: list):
    """Alinha os rótulos dos aspectos com os tokens
    das sub-words.

    Args:
        labels: List with labels.
        word_ids: Word index.

	Returns:
		List with labels aligned.
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels



def tokenize_and_align_labels(examples: dict) -> Dataset:
	"""Tokenize and align labels with subword tokens.

	Args:
		examples: Pre-token.

	Returns:
		Tokens with labels.
	"""
	tokenized_inputs = tokenizer(
		examples['tokens'],
		truncation=True,
		is_split_into_words=True,
	)
	all_labels = examples['aspect_tags']
	new_labels = []
	for i, labels in enumerate(all_labels):
		word_ids = tokenized_inputs.word_ids(i)
		new_labels.append(align_labels_with_tokens(labels, word_ids))
	tokenized_inputs['labels'] = new_labels
	return tokenized_inputs


def train_test_val_split(
	data: Dataset, 
	test_size: float, 
	val_size: float) -> DatasetDict:
	"""It separates the data into training, testing 
	and validation.

	Args:
		data: Dataset to split.
		test_size: Test data size.
		val_size: Validation data size.

	Returns:
		Dataset with train, test e validation set.
	"""

	train_data = data.train_test_split(
		test_size=test_size+val_size,
		shuffle=True
	)
	test_val_data = train_data['test'].train_test_split(
		test_size=val_size/(test_size+val_size),
		shuffle=True
	)
	return DatasetDict({
		'train': train_data['train'],
		'test': test_val_data['train'],
		'validation': test_val_data['test']
	})


def dataset_to_tf_dataset(
	data: DatasetDict,
	data_collator: DataCollatorForTokenClassification,
	columns=list,
	batch_size: float = 8):
	"""Converting to TF dataset format.

	Args:
		data: Dataset.
		data_collator: Data collator object.
		columns: Column names.
		batch_size: Batch size.

	Returns:
		Dataset in the format that TF expects.
	"""
	tf_dataset = {}
	splits = ['train', 'test', 'validation']
	for split in splits:
		tf_dataset[split] = data[split].to_tf_dataset(
			columns=columns,
			collate_fn=data_collator,
			shuffle=True,
			batch_size=8
		)
	return tf_dataset


def build_model(
	model_checkpoint: str,
	id2label: dict,
	label2id: dict,
	from_pt=None):
	"""Generates the model in TF format.

	Args:
		model_checkpoint: Model name to download from hugging face hub.
		id2label: Mapper of id to label.
		label2id: Mapper of label to id.
		from_pt: Indicates whether or not sensors are in pytorch format.
	
	Returns:
		Desired model.
	"""
	return TFAutoModelForTokenClassification.from_pretrained(
			model_checkpoint,
			id2label=id2label,
			label2id=label2id,
			from_pt=from_pt
		)


def evaluate_model(
    model: TFBertForTokenClassification, 
    test_data: Dataset,
    label_names: list) -> dict:
	"""Calculates model evaluation metrics.

	Args:
		model: Model object.
		test_data: Dataset to test the model.
    	label_names: Label names.

	Returs:
		Precision, recall and F1.
	"""

	# making predictions and computing metrics
	all_labels = []
	all_predictions = []
	metric = evaluate.load("seqeval")
	for batch in test_data:
	  logits = model.predict_on_batch(batch)["logits"]
	  labels = batch["labels"]
	  predictions = np.argmax(logits, axis=-1)
	  for prediction, label in zip(predictions, labels):
	      for predicted_idx, label_idx in zip(prediction, label):
	          if label_idx == -100:
	              continue
	          all_predictions.append(label_names[predicted_idx])
	          all_labels.append(label_names[label_idx])
	return metric.compute(
		predictions=[all_predictions], 
		references=[all_labels])

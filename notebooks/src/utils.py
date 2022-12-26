import evaluate

import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict

from sklearn.model_selection import StratifiedKFold

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

    Returns:
        Tokenizador carregado.
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
        test_size=test_size + val_size,
        shuffle=True
    )
    test_val_data = train_data['test'].train_test_split(
        test_size=val_size / (test_size + val_size),
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
        batch_size: int = 8):
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
            batch_size=batch_size
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


def aspect_counter(data: pd.DataFrame) -> pd.DataFrame:
    """Contabiliza o número de aspectos por registro.

        Args:
            data: Dataframe com os dados contendo, para cada registro, uma
            coluna 'aspect_tags' com os valores em formato de lista.

        Returns
            Dataframe com uma nova coluna, com o número de aspectos por registro.
        """
    num_aspect = []
    for record in data.aspect_tags.values:
        num_aspect.append(record.count(1))
    data['num_aspects'] = num_aspect
    return data


def has_aspect(data: pd.DataFrame) -> pd.DataFrame:
    """Define se um registro possui ou não aspecto.

        Args:
            data: Dataframe com os dados contendo, para cada registro, uma
            coluna 'num_aspects' com a quantidade de aspectos que cada registro
            possui.

        Returns:
            Dataframe com uma nova coluna, indicando se o registro possui ou
            não aspectos.
        """
    data['has_aspect'] = data.num_aspects > 0
    data['has_aspect'] = data['has_aspect'].astype(int)
    return data


def stratified_k_fold(data: pd.DataFrame,
                      X_cols: list,
                      y_col: str,
                      k: int = 10) -> pd.DataFrame:
    """Gera particões do dataframe com base em amostragem estratificada.

        Args:
            data: Dataframe com os dados a serem particionados.
            X_cols: Lista com os nomes das colunas preditoras.
            y_col: Nome da coluna referência para estraficacão.
            k: Número de particões.

        Returns:
            Dataframe com uma nova coluna 'fold' indicando a particão que o
            registro está alocado.
        """

    # seprando X e y
    y = data[y_col].values
    X = data[X_cols].values

    # gerando os folds
    new_data = pd.DataFrame()
    skf = StratifiedKFold(n_splits=k)
    for fold, (_, idx) in enumerate((skf.split(X, y))):
        curr = data.iloc[idx].copy()
        curr['fold'] = fold + 1
        new_data = pd.concat([new_data, curr]).reset_index(drop=True)
    new_data.fold = new_data.fold.astype(int)
    return new_data


# def tokenize_and_align_labels(examples: dict) -> Dataset:
#     """Tokenize and align labels with subword tokens.
#
# 	Args:
# 		examples: Pre-token.
#
# 	Returns:
# 		Tokens with labels.
# 	"""
#     tokenized_inputs = tokenizer(
#         examples['tokens'],
#         truncation=True,
#         is_split_into_words=True,
#     )
#     all_labels = examples['aspect_tags']
#     new_labels = []
#     for i, labels in enumerate(all_labels):
#         word_ids = tokenized_inputs.word_ids(i)
#         new_labels.append(align_labels_with_tokens(labels, word_ids))
#     tokenized_inputs['labels'] = new_labels
#     return tokenized_inputs

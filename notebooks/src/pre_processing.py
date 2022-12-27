import os

import pandas as pd

from datasets import Dataset
from src import utils


def pre_processing_tv_dataset(path: str) -> Dataset or pd.DataFrame:
    """Pre-processing TV base data. Generating token-compatible
    sequences with aspects.

    Args:
        path: TV base file path.

    Returns:
        Dataset with pre-processed data.
    """

    # reading the data
    df = utils.json_to_dataframe(path)

    # pre-processing the data
    data = {'tokens': [], 'aspect_tags': []}
    for idx, row in df.iterrows():

        # separating the tokens
        tokens = utils.split(row.tokens)

        # tagging the aspects in sequences (list)
        labels = [0] * len(tokens)
        for start, end in row['explicit aspects positions']:
            for i in range(end - start):
                tag = 1 if i == 0 else 2
                labels[start + i] = tag
        data['tokens'].append(tokens)
        data['aspect_tags'].append(labels)

    return pd.DataFrame(data)


def extract_tokens_and_aspects(path: str) -> pd.DataFrame:
    """Extracts tokens and aspects from the dataset.

    Args:
        path: Path of json file.

    Returns:
        Dataframe with tokens and aspects.
    """

    # Lendo o arquivo.
    data = utils.json_to_dataframe(path)

    # Extracting tokens and aspects.
    tokens_aspects = []
    for phrases in data['phrases']:
        for sentences in phrases:
            curr_tokens = []
            curr_aspects = []
            for word in sentences['words']:
                curr_tokens.append(word['plain'])
            for aspect in sentences['aspects']:
                curr_aspects.append(aspect['index'])
            tokens_aspects.append((curr_tokens, curr_aspects))
    return pd.DataFrame(
        tokens_aspects,
        columns=['tokens', 'aspect_tags'])


def _pre_processing_reli_dataset(path: str) -> pd.DataFrame:
    """Pre-processing of the books dataset.

    Args:
        path: Path of json file.

    Returns:
        Dataset com os dados prontos
    """

    # Extraindo os tokens e aspectos do arquivo json.
    data = extract_tokens_and_aspects(path)

    # Pré-processando os dados.
    aspect_tags = []
    for idx, row in data.iterrows():
        curr_aspect_tags = [0] * len(row['tokens'])
        if len(row['aspect_tags']) > 0:
            for aspect_idx in row['aspect_tags']:
                for i in range(len(aspect_idx)):
                    tag = 1 if i == 0 else 2
                    curr_aspect_tags[aspect_idx[i]] = tag
        aspect_tags.append(curr_aspect_tags)
    data['aspect_tags'] = aspect_tags
    return data


def split_author(string: str) -> str:
    return string.split('/')[-1].split('.')[0].split('_')[0]


def pre_processing_reli_dataset(path: str):
    """Pré-processamento do dataset livros.

    Args:
        path: Caminho dos arquivos json.

    Returns:
        Dataset with pre-processed data.
    """

    # listing the files.
    paths = [os.path.join(path, name) for name in os.listdir(path)]

    # pre-processing
    data = _pre_processing_reli_dataset(paths[0])
    data['author'] = split_author(paths[0])
    for path in paths[1:]:
        curr_dataset = _pre_processing_reli_dataset(path)
        curr_dataset['author'] = split_author(path)
        data = pd.concat([data, curr_dataset]).reset_index(drop=True)
    return data

import json
import pandas as pd

from transformers import AutoTokenizer


def json_to_dataframe(path: str) -> pd.DataFrame:
    """Transforma o arquivo de dados json em dataframe.

    Args:
        path: Caminho do arquivo json.

    Retuns:
        df: Dataframe com os dados.
    """
    wrk = pd.read_json('gdrive/My Drive/Datasets/data.json')
    return pd.json_normalize(wrk.reviews)


def split(text: str) -> list:
    """Separa a string de lista de string em lista de tokens.

    Args:
        text: String com a sentenća a ser separada.

    Returns:

    """
    return text.strip('][').split(', ')


def build_tokenizer(model_checkpoint: str):
    """Inicializa o tokenizador.

    Args:
        model_checkpoint: Nome do modelo.
    """
    return AutoTokenizer.from_pretrained(model_checkpoint)


def align_labels_with_tokens(labels: list, word_ids: list):
    """Alinha os rótulos dos aspectos com os tokens
    das sub-words.

    Args:
        labels: 
        word_ids:
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
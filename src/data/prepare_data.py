import json
import logging
import os
import re
from typing import List, Union

import datasets
import hydra
import pandas

# import phonemizer
import soundfile
from omegaconf import DictConfig
from torch.utils.data import Dataset

# Set environment variables for full trace of errors
os.environ["HYDRA_FULL_ERROR"] = "1"
data_path: str = os.path.join(os.environ.get("HF_HOME"), "../asr_vibravox/")
dir_path: str = str(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))

logger: logging.Logger = logging.getLogger(__name__)

# French vowels
vowels = ["i", "e", "ɛ", "a", "ɑ", "o", "ɔ", "u", "y", "ø", "œ", "ə"]
# tilde
tilde = ["̃"]
# Semi-vowels
semi_vowels = ["j", "w"]
# French consonants
consonants = [
    "p",
    "b",
    "t",
    "d",
    "k",
    "ɡ",
    "f",
    "v",
    "s",
    "z",
    "ʃ",
    "ʒ",
    "m",
    "n",
    "ɲ",
    "ŋ",
    "l",
    "ʁ",
]
# Other symbols
other_symbols = [" "]
# Complete list
french_phonetic_alphabet = vowels + tilde + semi_vowels + consonants + other_symbols


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    main downloads and extracts the dataset, splits them into {train, validation, test},
    removes special characters and transforms diacritics and ligatures into characters from the latin alphabet,
    saves the vocabulary in vocab.json,
    and filters each audio by its length.
    all data is cached by huggingface dataset in the shared cache folder defined by os.environ

    Args:
        cfg (`DictConfig`): The config file containing parameters for the Dataset.
    """
    # Read in the dataframe and split by training, test and validation splits
    ld_train = datasets.load_dataset(
        cfg.data_module.dataset_name,
        cfg.data_module.dataset_configuration,
        split=f"train.{str(cfg.data_module.number_of_hours_train)}"
        if not str(cfg.data_module.number_of_hours_train) == "-1"
        else "train",
    )
    ld_eval = datasets.load_dataset(
        cfg.data_module.dataset_name,
        cfg.data_module.dataset_configuration,
        split="validation",
    )
    ld_test = datasets.load_dataset(
        cfg.data_module.dataset_name,
        cfg.data_module.dataset_configuration,
        split="test",
    )

    if "vibravox" not in cfg.data_module.dataset_name:
        logger.info("Removing unwanted characters")
        f_remove_unwanted_characters = lambda batch: remove_unwanted_characters(
            batch, cfg.data_module.task_type
        )
        ld_train = ld_train.map(f_remove_unwanted_characters)
        ld_eval = ld_eval.map(f_remove_unwanted_characters)
        ld_test = ld_test.map(f_remove_unwanted_characters)

    def filter_dict_words(batch, list_str):
        for x in list_str:
            return not x in batch

    try:
        list_unwanted_words = cfg.list_unwanted_words
    except:
        list_unwanted_words = None
    if list_unwanted_words is not None:
        f_filter_dict_words = lambda batch: filter_dict_words(
            batch, list_unwanted_words
        )
        logger.info("Removing lines containing words one doesn't want")
        ld_train = ld_train.filter(f_filter_dict_words, input_columns=["text"])
        ld_eval = ld_eval.filter(f_filter_dict_words, input_columns=["text"])
        ld_test = ld_test.filter(f_filter_dict_words, input_columns=["text"])

    if (
        "vibravox" not in cfg.data_module.dataset_name
        and cfg.data_module.task_type == "phoneme"
    ):
        logger.info("Phonemizing texts")
        ld_train = ld_train.map(phonemize_characters, batched=True, batch_size=1_000)
        ld_eval = ld_eval.map(phonemize_characters, batched=True, batch_size=2_000)
        ld_test = ld_test.map(phonemize_characters, batched=True, batch_size=2_000)

        logger.info("Removing sentences with bad phonemes")
        filter_for_french_phonetic_alphabet = lambda x: all(
            char in french_phonetic_alphabet for char in x
        )
        ld_train = ld_train.filter(
            filter_for_french_phonetic_alphabet, input_columns=["phoneme"]
        )
        ld_eval = ld_eval.filter(
            filter_for_french_phonetic_alphabet, input_columns=["phoneme"]
        )
        ld_test = ld_test.filter(
            filter_for_french_phonetic_alphabet, input_columns=["phoneme"]
        )

        logger.info(
            "Filtering labels by length (keeping only phonemized labels with more than 4 phonemes in order to avoid NaNs in CTC loss)"
        )

        filter_phoneme_labels_by_length = lambda x: len(x) > 4
        ld_train = ld_train.filter(
            filter_phoneme_labels_by_length, input_columns=["phoneme"]
        )
        ld_eval = ld_eval.filter(
            filter_phoneme_labels_by_length, input_columns=["phoneme"]
        )
        ld_test = ld_test.filter(
            filter_phoneme_labels_by_length, input_columns=["phoneme"]
        )

    # Construct and save the vocab file
    save_vocab(
        ld_train,
        processor=cfg.data_module.processor,
        task_type=cfg.data_module.task_type,
    )

    if not cfg.data_module.max_input_length_in_sec == -1:  # if filter by length
        logger.info(
            f"Resampling dataset audios to {str(cfg.data_module.sampling_rate)} Hz"
        )
        ld_train = ld_train.cast_column(
            "audio", datasets.Audio(sampling_rate=cfg.data_module.sampling_rate)
        )
        ld_eval = ld_eval.cast_column(
            "audio", datasets.Audio(sampling_rate=cfg.data_module.sampling_rate)
        )
        ld_test = ld_test.cast_column(
            "audio", datasets.Audio(sampling_rate=cfg.data_module.sampling_rate)
        )

        logger.info(
            "Evaluating length of each audio and adding column to cached dataset"
        )

        logger.info("Filtering audios by length")

        if "vibravox" not in cfg.data_module.dataset_name:
            ld_train = ld_train.map(add_column_input_length)
            ld_eval = ld_eval.map(add_column_input_length)
            ld_test = ld_test.map(add_column_input_length)

            filter_by_length = lambda x: x < cfg.data_module.max_input_length_in_sec
            ld_train = ld_train.filter(filter_by_length, input_columns=["input_length"])
            ld_eval = ld_eval.filter(filter_by_length, input_columns=["input_length"])
            ld_test = ld_test.filter(filter_by_length, input_columns=["input_length"])

            filter_by_length = lambda x: x > cfg.data_module.min_input_length_in_sec
            ld_train = ld_train.filter(filter_by_length, input_columns=["input_length"])
            ld_eval = ld_eval.filter(filter_by_length, input_columns=["input_length"])
            ld_test = ld_test.filter(filter_by_length, input_columns=["input_length"])
        else:
            filter_by_length = lambda x: x < cfg.data_module.max_input_length_in_sec
            ld_train = ld_train.filter(filter_by_length, input_columns=["audio_length"])
            ld_eval = ld_eval.filter(filter_by_length, input_columns=["audio_length"])
            ld_test = ld_test.filter(filter_by_length, input_columns=["audio_length"])

            filter_by_length = lambda x: x > cfg.data_module.min_input_length_in_sec
            ld_train = ld_train.filter(filter_by_length, input_columns=["audio_length"])
            ld_eval = ld_eval.filter(filter_by_length, input_columns=["audio_length"])
            ld_test = ld_test.filter(filter_by_length, input_columns=["audio_length"])

        partial_path = f"{data_path}{cfg.data_module.dataset_name}_filtered/{cfg.data_module.language}/{str(int(cfg.data_module.max_input_length_in_sec))}/{cfg.data_module.dataset_configuration}"
    else:  # if no_filter
        partial_path = f"{data_path}{cfg.data_module.dataset_name}_filtered/{cfg.data_module.language}/no_filter/{cfg.data_module.dataset_configuration}"

    logger.info(f"Saving dataset to {partial_path}")
    ld_train.save_to_disk(f"{partial_path}/train")
    ld_eval.save_to_disk(f"{partial_path}/validation")
    ld_test.save_to_disk(f"{partial_path}/test")


def remove_unwanted_characters(
    batch: Union[pandas.DataFrame, Dataset], task_type: str = "phoneme"
) -> Union[pandas.DataFrame, Dataset]:
    """
    Removes special characters and transforms diacritics and ligatures into characters from the latin alphabet.

    Args:
        batch (`Union[pandas.DataFrame, Dataset]`): The dataset.
        task_type (str): Defaults to `phoneme`. Whether to do Speech-to-Text or Speech-to-Phoneme.
    Returns:
        `Union[pandas.DataFrame, Dataset]`: batch without special characters, diacritics and ligatures.
    """
    chars_to_remove_regex = '[\,\?\.\!\;\:"\(\)\#\d+]'

    # remove special characters
    try:
        batch["text"] = re.sub(chars_to_remove_regex, "", batch["text"]).lower().strip()
    except:
        try:
            batch["text"] = (
                re.sub(chars_to_remove_regex, "", batch["sentence"]).lower().strip()
            )
        except:
            batch["text"] = (
                re.sub(chars_to_remove_regex, "", batch["transcription"])
                .lower()
                .strip()
            )

    batch["text"] = re.sub("-", " ", batch["text"])
    batch["text"] = re.sub("_", " ", batch["text"])

    # remove annoying characters
    # diacritics
    if task_type == "text":
        batch["text"] = re.sub("â", "a", batch["text"])
        batch["text"] = re.sub("ā", "a", batch["text"])
        batch["text"] = re.sub("ä", "a", batch["text"])
        batch["text"] = re.sub("á", "a", batch["text"])
        batch["text"] = re.sub("ȧ", "a", batch["text"])
        batch["text"] = re.sub("à", "a", batch["text"])
        batch["text"] = re.sub("ã", "a", batch["text"])

        # batch["text"] = re.sub("é", "e", batch["text"]) # we keep 'é' because it has pronunciation usefulness
        # batch["text"] = re.sub("è", "e", batch["text"]) # we keep 'è' because it has pronunciation usefulness
        # batch["text"] = re.sub("ê", "e", batch["text"]) # we keep 'ê' because it has pronunciation usefulness
        batch["text"] = re.sub("ë", "e", batch["text"])
        batch["text"] = re.sub("ę", "e", batch["text"])

        batch["text"] = re.sub("ï", "i", batch["text"])
        batch["text"] = re.sub("î", "i", batch["text"])
        batch["text"] = re.sub("i", "i", batch["text"])  # i point suscrit
        batch["text"] = re.sub("í", "i", batch["text"])
        batch["text"] = re.sub("ī", "i", batch["text"])

        batch["text"] = re.sub("j", "j", batch["text"])  # j point suscrit

        batch["text"] = re.sub("ł", "l", batch["text"])

        batch["text"] = re.sub("ô", "o", batch["text"])
        batch["text"] = re.sub("ó", "o", batch["text"])
        batch["text"] = re.sub("ø", "o", batch["text"])
        batch["text"] = re.sub("ō", "o", batch["text"])
        batch["text"] = re.sub("õ", "o", batch["text"])
        batch["text"] = re.sub("ö", "o", batch["text"])

        batch["text"] = re.sub("š", "s", batch["text"])
        batch["text"] = re.sub("ṣ", "s", batch["text"])

        batch["text"] = re.sub("ṭ", "t", batch["text"])

        batch["text"] = re.sub("ù", "u", batch["text"])
        batch["text"] = re.sub("û", "u", batch["text"])
        batch["text"] = re.sub("ü", "u", batch["text"])
        batch["text"] = re.sub("ú", "u", batch["text"])

        batch["text"] = re.sub("ÿ", "y", batch["text"])
        batch["text"] = re.sub("y̌", "y", batch["text"])

        # batch["text"] = re.sub("ç", "c", batch["text"]) # we keep 'ç' because it has pronunciation usefulness
        batch["text"] = re.sub("č", "c", batch["text"])

        batch["text"] = re.sub("ñ", "n", batch["text"])
        batch["text"] = re.sub("ṇ", "n", batch["text"])

        batch["text"] = re.sub("ẓ", "z", batch["text"])
        batch["text"] = re.sub("ž", "z", batch["text"])

        # ligatures
        batch["text"] = re.sub("æ", "ae", batch["text"])
        batch["text"] = re.sub("œ", "oe", batch["text"])

        batch["text"] = re.sub("ꜳ", "aa", batch["text"])
        batch["text"] = re.sub("ꜵ", "ao", batch["text"])
        batch["text"] = re.sub("ꜷ", "au", batch["text"])
        batch["text"] = re.sub("ꜹ", "av", batch["text"])
        batch["text"] = re.sub("ꜽ", "ay", batch["text"])
        batch["text"] = re.sub("ȸ", "db", batch["text"])
        batch["text"] = re.sub("ʣ", "dz", batch["text"])
        batch["text"] = re.sub("ﬀ", "ff ", batch["text"])
        batch["text"] = re.sub("ﬁ", "fi", batch["text"])
        batch["text"] = re.sub("ﬂ", "fl", batch["text"])
        batch["text"] = re.sub("ﬃ", "ffi", batch["text"])
        batch["text"] = re.sub("ﬄ", "ffl", batch["text"])
        batch["text"] = re.sub("ĳ", "ij", batch["text"])
        batch["text"] = re.sub("ǉ", "lj", batch["text"])
        batch["text"] = re.sub("ǌ", "nj", batch["text"])
        batch["text"] = re.sub("ꝏ", "oo", batch["text"])
        batch["text"] = re.sub("ȹ", "qp", batch["text"])
        batch["text"] = re.sub("ﬆ", "st", batch["text"])
        batch["text"] = re.sub("ﬅ", "ft", batch["text"])
        batch["text"] = re.sub("ʦ", "ts", batch["text"])
        batch["text"] = re.sub("ᵫ", "ue", batch["text"])
        batch["text"] = re.sub("ꭣ", "uo", batch["text"])
        batch["text"] = re.sub("ꝡ", "vy", batch["text"])

    return batch


def phonemize_characters(
    batch: Union[pandas.DataFrame, Dataset]
) -> Union[pandas.DataFrame, Dataset]:
    """
    Phonemizes text i.e. translates french strings to international phoneme alphabet french strings
    e.g. "Enchanté, je m'appelle Malo" -> "ãʃãte ʒə mapɛl malo"

    Args:
        batch (`Union[pandas.DataFrame, Dataset]`): The dataset.
    Returns:
        `Union[pandas.DataFrame, Dataset]`: The dataset but instead of french texts, it is phonemized french text.
    """
    backend = phonemizer.backend.EspeakBackend(
        language="fr-fr", language_switch="remove-utterance"
    )
    batch["phoneme"] = backend.phonemize(batch["text"], strip=True, njobs=1)

    return batch


def save_vocab(
    dataset: Union[pandas.DataFrame, Dataset],
    processor: str,
    task_type: str = "phoneme",
) -> None:
    """
    Saves the processed vocab file as 'vocab.json', to be ingested by tokenizer.

    Args:
        dataset (`Union[pandas.DataFrame, Dataset]`): The dataset.
        processor (str): path to processor config
        task_type (str): Defaults to `phoneme`. Whether to do Speech-to-Text or Speech-to-Phoneme.
    Returns:
        NoneType: None
    """
    vocab = construct_vocab(dataset[task_type])

    print(f"dataset['text'] = {dataset['text'][:10]}")
    if task_type == "phoneme":
        print(f"dataset['phoneme'] = {dataset['phoneme'][:10]}")

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
    vocab_dict["|"] = vocab_dict[" "]
    _ = vocab_dict.pop(" ")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(f"{dir_path}{processor}/vocab.json", "w", encoding="utf-8") as fl:
        json.dump(vocab_dict, fl, ensure_ascii=False)

    logger.info("Created Vocab file!")


def construct_vocab(texts: str) -> List[str]:
    """
    Get unique characters from all the text in a list.

    Args:
        texts (str): The texts.
    Returns:
        List[str]: a list of texts.
    """
    all_text = " ".join(texts)
    vocab = list(set(all_text))
    return vocab


def add_column_input_length(
    batch: Union[pandas.DataFrame, Dataset]
) -> Union[pandas.DataFrame, Dataset]:
    """
    Adds a column named 'input_length' in batch.

    Args:
        batch (`Union[pandas.DataFrame, Dataset]`): The dataset on which the function adds a new column.
    Returns:
        `Union[pandas.DataFrame, Dataset]`: batch but with the new added column 'input_length' in seconds
    """
    try:
        batch["input_length"] = soundfile.info(batch["file"]).duration
    except:
        batch["input_length"] = soundfile.info(batch["path"]).duration
    return batch


if __name__ == "__main__":
    main()

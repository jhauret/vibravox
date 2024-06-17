"""
This script is used to test all phonemizers on all microphones, thus creating a 6x6 matrix of the Phoneme Error Rate (PER).
"""
from typing import List, Tuple
from collections import Counter
import torchmetrics
import transformers
from datasets import load_dataset, Audio
import torch
from transformers import AutoProcessor, AutoModelForCTC
from tqdm import tqdm
import Levenshtein
import pickle

MICROPHONES = ["headset_microphone",
               "forehead_accelerometer",
               "rigid_in_ear_microphone",
               "soft_in_ear_microphone",
               "throat_microphone",
               "temple_vibration_pickup"]

PHONEMIZERS = [f"phonemizer_{microphone}" for microphone in MICROPHONES]


SAMPLE_RATE = 16_000
DATASETS = ["Cnam-LMSSC/vibravox2"]  #"Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp",
FEATURE_EXTRACTOR = transformers.Wav2Vec2FeatureExtractor()
TOKENIZER = transformers.Wav2Vec2CTCTokenizer.from_pretrained("Cnam-LMSSC/vibravox-phonemes-tokenizer")
PER = torchmetrics.text.CharErrorRate()

per_results = torch.empty((len(DATASETS), len(MICROPHONES), len(PHONEMIZERS)))
editops_occurrences_results = {}


def decode_operations(predicted_chr: str,
                      label_chr: str,
                      editops: List[Tuple[str, int, int]]) -> List[Tuple[str, str, str]]:
    """
    Decode the operations based on the edit operations.

    Args:
        predicted_chr (str): The predicted character.
        label_chr (str): The label character.
        editops (List[Tuple[str, int, int]]): The list of edit operations.

    Returns:
        List[Tuple[str, str, str]]: The list of decoded operations.
    """
    ops = []
    for editop in editops:
        op, pred_idx, label_idx = editop

        if op == "insert":
            label_token = label_chr[label_idx]
            ops.append((op, label_token, label_token))
        elif op == "delete":
            pred_token = predicted_chr[pred_idx]
            ops.append((op, pred_token, pred_token))
        else:
            label_token = label_chr[label_idx]
            pred_token = predicted_chr[pred_idx]
            ops.append((op, pred_token, label_token))

    return ops


def get_space_indices(string: str) -> List[int]:
    """
    Get the positions of spaces in a string.

    Args:
        string (str): The input string.

    Returns:
        List[int]: The list of space indices.
    """
    return [i for i, x in enumerate(string) if x == ' ']


def split_editops(pred: str,
                  target: str,
                  editops: List[Tuple[str, int, int]])\
        -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    """
    Split the edit operations into three categories: before space, in word, and all.

    Args:
        pred (str): The predicted string.
        target (str): The target string.
        editops (List[Tuple[str, int, int]]): The list of edit operations.

    Returns:
        Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]], List[Tuple[str, int, int]]]: The split edit operations.
    """
    pred_space_idx = get_space_indices(pred)
    target_space_idx = get_space_indices(target)

    raw_editops_before_space = []
    raw_editops_in_word = []
    for editop in editops:
        op, pred_idx, label_idx = editop

        if ((op == 'replace' and ((pred_idx+1) in pred_space_idx or (label_idx + 1) in target_space_idx)) or
            (op == 'delete' and (pred_idx+1) in pred_space_idx) or
            (op == 'insert' and (label_idx + 1) in target_space_idx)):
            raw_editops_before_space.append(editop)
        else:
            raw_editops_in_word.append(editop)

    return raw_editops_before_space, raw_editops_in_word, editops


for dataset_idx, dataset_name in enumerate(DATASETS):
    for microphone_idx, microphone in enumerate(MICROPHONES):
        test_dataset = load_dataset(dataset_name, "speech_clean", split="test", streaming=False)
        test_dataset = test_dataset.cast_column(f"audio.{microphone}", Audio(sampling_rate=SAMPLE_RATE, mono=False))
        for phonemizer_idx, phonemizer in enumerate(PHONEMIZERS):
            processor = AutoProcessor.from_pretrained(f"Cnam-LMSSC/{phonemizer}")
            model = AutoModelForCTC.from_pretrained(f"Cnam-LMSSC/{phonemizer}")
            preds, targets, editops_in_word, editops_before_space, editops_all = [], [], [], [], []
            for sample in tqdm(test_dataset):

                target_transcription = sample["phonemized_text"]

                # Compute predicted transcription
                audio_16kHz = sample[f"audio.{microphone}"]["array"]
                inputs = processor(audio_16kHz, sampling_rate=16_000, return_tensors="pt")
                logits = model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_transcription = processor.batch_decode(predicted_ids)[0]

                # Get edit operations
                raw_editops = Levenshtein.editops(predicted_transcription, target_transcription)
                raw_editops_before_space, raw_editops_in_word, raw_editops_all = split_editops(predicted_transcription, target_transcription, raw_editops)

                decoded_operations_in_word = decode_operations(predicted_transcription, target_transcription, raw_editops_in_word)
                decoded_operations_before_space = decode_operations(predicted_transcription, target_transcription, raw_editops_before_space)
                decoded_operations_all = decode_operations(predicted_transcription, target_transcription, raw_editops_all)

                preds.append(predicted_transcription)
                targets.append(target_transcription)

                editops_in_word += decoded_operations_in_word
                editops_before_space += decoded_operations_before_space
                editops_all += decoded_operations_all

            # Save PER
            per = PER(preds, targets)
            print(f"Test PER of {phonemizer} on {microphone} for {dataset_name} subset: {per * 100:.2f}%")
            per_results[dataset_idx, microphone_idx, phonemizer_idx] = per

            # Save edit operations
            occurrences_in_word = Counter(editops_in_word)
            occurrences_in_word = sorted(occurrences_in_word.items(), key=lambda x: x[1], reverse=True)
            occurrences_before_space = Counter(editops_before_space)
            occurrences_before_space = sorted(occurrences_before_space.items(), key=lambda x: x[1], reverse=True)
            occurrences_dict_all = Counter(editops_all)
            occurrences_dict_all = sorted(occurrences_dict_all.items(), key=lambda x: x[1], reverse=True)

            editops_occurrences_results[f"{dataset_name}-{microphone}-{phonemizer}-in_word"] = occurrences_in_word
            editops_occurrences_results[f"{dataset_name}-{microphone}-{phonemizer}-before_space"] = occurrences_before_space
            editops_occurrences_results[f"{dataset_name}-{microphone}-{phonemizer}-all"] = occurrences_dict_all


# Save results
torch.save(per_results, "./outputs/scripts/stp_phonemizer_per_results.pt")
with open('./outputs/scripts/stp_editops_occurrences_results.pickle', 'wb') as f:
    pickle.dump(editops_occurrences_results, f)

"""
This script is used to test all phonemizers on all microphones, thus creating a 6x6 matrix of the Phoneme Error Rate (PER).
"""

from collections import Counter
import torchmetrics
import transformers
from datasets import load_dataset, Audio
import torch
from transformers import AutoProcessor, AutoModelForCTC
from tqdm import tqdm
from vibravox.utils import split_editops, decode_operations
import Levenshtein
import pickle
from torchmetrics.functional.text import char_error_rate

MICROPHONES = ["headset_microphone",
               "forehead_accelerometer",
               "soft_in_ear_microphone",
               "rigid_in_ear_microphone",
               "throat_microphone",
               "temple_vibration_pickup"]

PHONEMIZERS = [f"phonemizer_{microphone}" for microphone in MICROPHONES]
SAMPLE_RATE = 16_000
DATASETS = ["Cnam-LMSSC/vibravox", "Cnam-LMSSC/vibravox_enhanced_by_EBEN"]
CUDA_IS_AVAILABLE = torch.cuda.is_available()

FEATURE_EXTRACTOR = transformers.Wav2Vec2FeatureExtractor()
TOKENIZER = transformers.Wav2Vec2CTCTokenizer.from_pretrained("Cnam-LMSSC/vibravox-phonemes-tokenizer")

per_results = torch.empty((len(DATASETS), len(MICROPHONES), len(PHONEMIZERS)))
editops_occurrences_results = {}

for dataset_idx, dataset_name in enumerate(DATASETS):
    for microphone_idx, microphone in enumerate(MICROPHONES):
        test_dataset = load_dataset(dataset_name, "speech_clean", split="test", streaming=False)
        test_dataset = test_dataset.cast_column(f"audio.{microphone}", Audio(sampling_rate=SAMPLE_RATE, mono=False))
        test_dataset = test_dataset.with_format("torch")
        for phonemizer_idx, phonemizer in enumerate(PHONEMIZERS):
            processor = AutoProcessor.from_pretrained(f"Cnam-LMSSC/{phonemizer}")
            model = AutoModelForCTC.from_pretrained(f"Cnam-LMSSC/{phonemizer}")
            if CUDA_IS_AVAILABLE:
                model.cuda()
            preds, targets, editops_in_word, editops_before_space, editops_all = [], [], [], [], []
            for sample in tqdm(test_dataset):

                target_transcription = sample["phonemized_text"]

                # Compute predicted transcription
                audio_16kHz = sample[f"audio.{microphone}"]["array"]
                inputs = processor(audio_16kHz, sampling_rate=16_000, return_tensors="pt")
                if CUDA_IS_AVAILABLE:
                    inputs.input_values = inputs.input_values.cuda()
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
            per = char_error_rate(preds, targets)
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

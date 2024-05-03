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

MICROPHONES = ["airborne.mouth_headworn.reference_microphone",
               "body_conducted.in_ear.rigid_earpiece_microphone",
               "body_conducted.in_ear.comply_foam_microphone",
               "body_conducted.forehead.miniature_accelerometer",
               "body_conducted.throat.piezoelectric_sensor",
               "body_conducted.temple.contact_microphone"]

PHONEMIZERS = [f"phonemizer_{microphone}" for microphone in MICROPHONES]
SAMPLE_RATE = 16_000
SUBSETS = ["speech_clean", "speech_noisy"]
FEATURE_EXTRACTOR = transformers.Wav2Vec2FeatureExtractor()
TOKENIZER = transformers.Wav2Vec2CTCTokenizer.from_pretrained("Cnam-LMSSC/vibravox-phonemes-tokenizer")
PER = torchmetrics.text.CharErrorRate()

per_results = torch.empty((len(SUBSETS), len(MICROPHONES), len(PHONEMIZERS)))
editops_occurrences_results = {}


def decode_operations(predicted_chr: str,
                      label_chr: str,
                      editops: List[Tuple[str, int, int]]) -> List[Tuple[str, str, str]]:
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


for subset_idx, subset_name in enumerate(SUBSETS):
    for microphone_idx, microphone in enumerate(MICROPHONES):
        test_dataset = load_dataset("Cnam-LMSSC/vibravox", subset_name, split="test", streaming=False)
        test_dataset = test_dataset.cast_column(f"audio.{microphone}", Audio(sampling_rate=SAMPLE_RATE, mono=False))
        for phonemizer_idx, phonemizer in enumerate(PHONEMIZERS):
            processor = AutoProcessor.from_pretrained(f"Cnam-LMSSC/{phonemizer}")
            model = AutoModelForCTC.from_pretrained(f"Cnam-LMSSC/{phonemizer}")
            preds, targets, editops = [], [], []
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
                decoded_editops = decode_operations(predicted_transcription, target_transcription, raw_editops)

                preds.append(predicted_transcription)
                targets.append(target_transcription)
                editops += decoded_editops

            # Save PER
            per = PER(preds, targets)
            print(f"Test PER of {phonemizer} on {microphone} for {subset_name} subset: {per * 100:.2f}%")
            per_results[subset_idx, microphone_idx, phonemizer_idx] = per

            # Save edit operations
            occurrences_dict = Counter(editops)
            occurrences_dict = sorted(occurrences_dict.items(), key=lambda x: x[1], reverse=True)
            editops_occurrences_results[f"{subset_name}-{microphone}-{phonemizer}"] = occurrences_dict

# Save results
torch.save(per_results, "./outputs/scripts/stp_phonemizer_per_results.pt")
with open('./outputs/scripts/stp_editops_occurrences_results.pickle', 'wb') as f:
    pickle.dump(editops_occurrences_results, f)

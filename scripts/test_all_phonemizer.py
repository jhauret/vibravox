"""
This script is used to test all phonemizers on all microphones, thus creating a 6x6 matrix of the Phoneme Error Rate (PER).
"""
import torchmetrics
import transformers
from datasets import load_dataset, Audio
import torch
from transformers import AutoProcessor, AutoModelForCTC
from tqdm import tqdm

MICROPHONES = ["body_conducted.temple.contact_microphone",
               "body_conducted.throat.piezoelectric_sensor",
               "body_conducted.in_ear.rigid_earpiece_microphone",
               "body_conducted.in_ear.comply_foam_microphone",
               "body_conducted.forehead.miniature_accelerometer",
               "airborne.mouth_headworn.reference_microphone"]
PHONEMIZERS = [f"phonemizer_{microphone}" for microphone in MICROPHONES]
SAMPLE_RATE = 16_000
SUBSETS = ["speech_clean", "speech_noisy"]
FEATURE_EXTRACTOR = transformers.Wav2Vec2FeatureExtractor()
TOKENIZER = transformers.Wav2Vec2CTCTokenizer.from_pretrained("Cnam-LMSSC/vibravox-phonemes-tokenizer")
PER = torchmetrics.text.CharErrorRate()

results = torch.empty((len(SUBSETS), len(MICROPHONES), len(PHONEMIZERS)))

for subset_idx, subset_name in enumerate(SUBSETS):
    for microphone_idx, microphone in enumerate(MICROPHONES):
        test_dataset = load_dataset("Cnam-LMSSC/vibravox", subset_name, split="test", streaming=False)
        test_dataset = test_dataset.cast_column(f"audio.{microphone}", Audio(sampling_rate=SAMPLE_RATE, mono=False))
        for phonemizer_idx, phonemizer in enumerate(PHONEMIZERS):
            processor = AutoProcessor.from_pretrained(f"Cnam-LMSSC/{phonemizer}")
            model = AutoModelForCTC.from_pretrained(f"Cnam-LMSSC/{phonemizer}")
            preds, targets = [], []
            for sample in tqdm(test_dataset):
                audio_16kHz = sample[f"audio.{microphone}"]["array"]
                inputs = processor(audio_16kHz, sampling_rate=16_000, return_tensors="pt")
                logits = model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]
                preds.append(transcription)
                targets.append(sample["phonemized_text"])

            per = PER(preds, targets)
            print(f"Test PER of {phonemizer} on {microphone} for {subset_name} subset: {per * 100:.2f}%")
            results[subset_idx, microphone_idx, phonemizer_idx] = per

print(results)
torch.save(results, "phonemizer_per_results.pt")

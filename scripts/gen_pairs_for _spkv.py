"""Generates pairs of utterances for speaker verification and then saves them in a pickle file."""

import pickle
import itertools
import random
import math
from lightning import seed_everything
from datasets import load_dataset
from pathlib import Path


seed_everything(42, workers=True)


DATASET_NAME = "Cnam-LMSSC/vibravox_enhanced_by_EBEN_dummy"  # For tests only, replace it by vibravox later

# Loading the dataset
dataset_dict = load_dataset(DATASET_NAME, "speech_clean", split="test", streaming=False)

# Only keep the "speaker_id" column with is the only column we need to generate pairs
dataset_dict = dataset_dict.select_columns(["speaker_id"])

# Order by speaker_id for easier pairing of audios :
dataset_dict = dataset_dict.sort("speaker_id")

# Calculate the number of utterances for each speaker and total number of speakers
utterances = []
nb_speakers = 0

for speaker in sorted(set(dataset_dict["speaker_id"])):
    num_rows = dataset_dict.filter(lambda example: example['speaker_id'] == speaker).num_rows
    utterances.append(num_rows)
    nb_speakers += 1

# Find the minimum number of utterances
min_utterances = min(utterances)
utterances.insert(0, 0)

# Calculate the accumulated utterances
accumulated_utterances = list(itertools.accumulate(utterances))

# Remove unneeded last element of accumulated_utterances :
del accumulated_utterances[-1]

# Generate the ranges for each speaker with min_utterances added
ranges_per_speaker = [list(range(accumulated_utterances[i], accumulated_utterances[i] + min_utterances)) for i in range(nb_speakers)]

# Initialize lists to store pairs
same_speakers_pairs = []
different_speakers_pairs = []

# Generate pairs for same and different speakers
for speaker in range(nb_speakers):
    # Same speaker pairs
    speaker_indices = ranges_per_speaker[speaker][:]
    same_speaker_combinations = list(itertools.combinations(speaker_indices, r=2))

    # Add the same speaker combinations to the list of same speaker pairs
    same_speakers_pairs += same_speaker_combinations

    # Different speaker pairs
    # Generate random indices for different speakers
    speaker_indices = [i for i in range(nb_speakers) if i != speaker]
    target_indices = random.choices(ranges_per_speaker[speaker], k=math.comb(min_utterances, 2))
    other_indices = list(zip(
        random.choices(speaker_indices, k=math.comb(min_utterances, 2)),
        random.choices(range(min_utterances), k=math.comb(min_utterances, 2))))

    # Get the corresponding indices from ranges_per_speaker
    other = [ranges_per_speaker[other_idx[0]][other_idx[1]] for other_idx in other_indices]

    # Combine target and other indices to form pairs and add to the list of different speaker pairs
    different_speakers_pairs += list(zip(target_indices, other))

# Combine same and different speaker pairs
total_pairs = same_speakers_pairs + different_speakers_pairs
# Save total_pairs list of pairs to a pickle file in the same directory :

with open(Path(__file__).parent.parent / "lightning_datamodule/misc/pairs.pkl", 'wb') as f:
    pickle.dump(total_pairs, f)

print("Done !")
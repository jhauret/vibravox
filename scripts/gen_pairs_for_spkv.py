"""Generates pairs of utterances for speaker verification and then saves them in a pickle file."""

import pickle
import itertools
import random
import math
from lightning import seed_everything
from datasets import load_dataset
from pathlib import Path

def load_and_process_data(dataset_name):

    """
    Load and process data for a specified dataset to generate pairs based on "speaker_id" and "gender".
    Parameters:
        dataset_name (str): The name of the dataset to load and process.
    Returns:
        Dataset: A processed dataset containing only "speaker_id" and "gender" columns sorted by "speaker_id".
    """

    dataset_dict = load_dataset(dataset_name, "speech_clean", split="test", streaming=False)
    # Only keep the "speaker_id" and "gender" columns which are the only columns we need to generate pairs
    dataset_dict = dataset_dict.select_columns(["speaker_id", "gender"])
    dataset_dict = dataset_dict.sort("speaker_id")
    return dataset_dict

def generate_ranges_per_speaker(dataset_dict):

    """
    Calculate the number of utterances for each speaker and total number of speakers.

    :param dataset_dict: A dictionary containing the dataset with speaker_ids and genders
    :return: A tuple containing ranges_per_speaker:List[List[int]], nb_speakers:int , and min_utterances:int
    """
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

    # Remove unneeded last element of accumulated_utterances
    del accumulated_utterances[-1]

    # Generate the ranges for each speaker with min_utterances added
    ranges_per_speaker = [list(range(accumulated_utterances[i], accumulated_utterances[i] + min_utterances))
                          for i in range(nb_speakers)]
    return ranges_per_speaker, nb_speakers, min_utterances

def get_gender_per_speaker(dataset_dict,ranges_per_speaker,nb_speakers):

    """
    Generate the indices of male and female speakers in the dataset.

    Parameters:
    - dataset_dict (dict): The dictionary containing speaker information.
    - ranges_per_speaker List[List[int]]: List of ranges per speaker.
    - nb_speakers (int): Number of speakers in the dataset.

    Returns:
    - males_speaker_idx (list): List of indices of male speakers.
    - females_speaker_idx (list): List of indices of female speakers.
    """

    males_speaker_idx = []
    females_speaker_idx = []
    for i in range(nb_speakers):
        if dataset_dict["gender"][ranges_per_speaker[i][0]] == "male":
            males_speaker_idx.append(i)
        else:
            females_speaker_idx.append(i)
    return males_speaker_idx, females_speaker_idx


def generate_speaker_pairs(ranges_per_speaker, nb_speakers, min_utterances):

    """
        Generate pairs of same and different speakers based on the provided parameters.

        Parameters:
            ranges_per_speaker List[List[int]]: A list containing ranges of utterances for each speaker.
            nb_speakers (int): The number of speakers.
            min_utterances (int): The minimum number of utterances.
        Returns:
            List[Tuple[int, int]]: A list of pairs representing same and different speaker pairs.
        """

    # Initialize lists to store pairs
    same_speakers_pairs = []
    different_speakers_pairs = []

    # Generate pairs for same and different speakers
    for speaker in range(nb_speakers):
        # Same speaker pairs
        speaker_indices = ranges_per_speaker[speaker][:]
        same_speaker_combinations = list(itertools.combinations(speaker_indices, r=2))
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

    return total_pairs


def generate_speaker_pairs_same_gender(ranges_per_speaker, males_speaker_idx, females_speaker_idx, min_utterances):

    """
    A function to generate speaker pairs with the same gender, considering minimum utterances.

    Parameters:
    - ranges_per_speaker List[List[int]]: A list containing ranges of utterances for each speaker.
    - males_speaker_idx: a list of indices corresponding to male speakers
    - females_speaker_idx: a list of indices corresponding to female speakers
    - min_utterances: an integer representing the minimum number of utterances

    Returns:
    - total_pairs_same_gender List[Tuple[int, int]]: a list of speaker pairs with the same gender based on the conditions
    """

    # Initialize lists to store pairs
    same_speakers_same_gender_pairs = []
    different_speakers_same_gender_pairs = []

    genders = ["male","female"]

    # Generate pairs for same and different speakers sharing the same gender

    for gender in genders:
        if gender == "male":
            speakers = males_speaker_idx
        else:
            speakers = females_speaker_idx
        for speaker in speakers:
            # Same speaker pairs, same gender :
            speaker_indices = ranges_per_speaker[speaker][:]
            same_speaker_combinations = list(itertools.combinations(speaker_indices, r=2))
            same_speakers_same_gender_pairs += same_speaker_combinations

            # Different speaker pairs
            # Generate random indices for different speakers
            speaker_indices = [i for i in speakers if i != speaker]
            target_indices = random.choices(ranges_per_speaker[speaker], k=math.comb(min_utterances, 2))
            other_indices = list(zip(
                random.choices(speaker_indices, k=math.comb(min_utterances, 2)),
                random.choices(range(min_utterances), k=math.comb(min_utterances, 2))))

            # Get the corresponding indices from ranges_per_speaker
            other = [ranges_per_speaker[other_idx[0]][other_idx[1]] for other_idx in other_indices]

            # Combine target and other indices to form pairs and add to the list of different speaker pairs
            different_speakers_same_gender_pairs += list(zip(target_indices, other))

    # Combine same and different speaker pairs
    total_pairs_same_gender = same_speakers_same_gender_pairs + different_speakers_same_gender_pairs

    return total_pairs_same_gender

def save_pairs_to_pickle(total_pairs,filename):

    """
    Save pairs to a pickle file.

    Args:
        total_pairs (list): List of pairs to be saved.
        filename (str): Name of the pickle file (with extension .pkl) to save the pairs to.

    Returns:
        None
    """

    with open(Path(__file__).parent.parent / "configs/lightning_datamodule/spkv_pairs" / filename, 'wb') as f:
        pickle.dump(total_pairs, f)



if __name__ == "__main__":
    seed_everything(42, workers=True) # For deterministic picking of pairs and reproducibility
    DATASET_NAME = "Cnam-LMSSC/vibravox"
    dataset_dict = load_and_process_data(dataset_name = DATASET_NAME)
    ranges_per_speaker, nb_speakers, min_utterances = generate_ranges_per_speaker(dataset_dict)
    males_speaker_idx, females_speaker_idx = get_gender_per_speaker(dataset_dict, ranges_per_speaker,nb_speakers)
    total_pairs = generate_speaker_pairs(ranges_per_speaker, nb_speakers, min_utterances)
    total_pairs_same_gender = generate_speaker_pairs_same_gender(ranges_per_speaker, males_speaker_idx, females_speaker_idx, min_utterances)
    print(len(total_pairs))
    save_pairs_to_pickle(total_pairs, "pairs.pkl")
    save_pairs_to_pickle(total_pairs_same_gender, "pairs_same_gender.pkl")
    print("Pairs generated and saved to pickle files in", Path(__file__).parent.parent / "configs/lightning_datamodule/spkv_pairs")
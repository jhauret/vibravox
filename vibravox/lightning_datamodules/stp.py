from datasets import load_dataset, Audio
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer


class STPLightningDataModule(LightningDataModule):

    DATASET_NAME = "Cnam-LMSSC/vibravox"
    LANGUAGE = "fr"

    def __init__(
        self,
        sample_rate: int = 16000,
        subset_name: str = "bwe_in-ear_rigid_earpiece_microphone",
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        min_duration: float = 0.0,
        max_duration: float = -1.0,
    ):

        """
        LightningDataModule for Speech-to-Phoneme (STP).

        Args:
            sample_rate (int, optional): Sample rate of the audio files. Defaults to 16000.
            subset_name (str, optional): Name of the configuration. Defaults to "BWE_In-ear_Comply_Foam_microphone".
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            min_duration (float): Minimum duration of the audio files in seconds. Smaller files are removed. Defaults to 0.0.
            max_duration (float): Maximum duration of the audio files in seconds. Longer files are removed. Defaults to -1.0.
                                  -1.0 means that the audios are not filtered.
        """

        super().__init__()

        self.sample_rate = sample_rate
        self.subset_name = subset_name
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.audio_processor = Wav2Vec2FeatureExtractor()
        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file='/home/julien/Bureau/github/vibravox/configs/lightning_datamodule/tokenizer_vocab/minimal_vocab.json')
        #self.processor = Wav2Vec2Processor.from_pretrained("Cnam-LMSSC/wav2vec2-french-phonemizer")
        # Note: do we really want to normalize the audio? (
        # Cnam-LMSSC/wav2vec2-french-phonemizer/wav2vec2-french-phonemizer/preprocessor_config.json -> "normalize": True

    def setup(self, stage=None):

        datasets = load_dataset(
            self.DATASET_NAME, self.subset_name, streaming=self.streaming
        )

        datasets = datasets.select_columns(["audio", "phonemes"])

        # Resample the audio to the right sample rate
        datasets = datasets.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate, mono=False)
        )

        #datasets = datasets.with_format("torch")

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def data_collator(self, batch):
        """
        Custom data collator function to dynamically pad the data.

        Args:
            batch
        Returns:
            processed_batch
        """

        audios = [sample["audio"]["array"] for sample in batch]
        phonemes = [sample["phonemes"] for sample in batch]

        audios_processed = self.audio_processor(
            raw_speech=audios,
            padding='longest',
            return_tensors="pt",
            sampling_rate=self.sample_rate,  # Do not resample anything, simple verification
            pad_to_multiple_of=128,  # Because NVIDIA GeForce RTX 2080 Ti have 128 Concurrent Kernel Execution
        ).input_values

        # phonemes_processed_bis = self.processor(
        #     text=phonemes,
        #     padding=True,
        #     return_tensors="pt",
        #     pad_to_multiple_of=128,
        #     # Because NVIDIA GeForce RTX 2080 Ti have 128 Concurrent Kernel Execution
        # )

        phonemes_processed = self.tokenizer(
            text=phonemes,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=128,
            # Because NVIDIA GeForce RTX 2080 Ti have 128 Concurrent Kernel Execution
        )

        # Replace padding with -100 to ignore loss correctly
        phonemes_processed = phonemes_processed["input_ids"].masked_fill(
            phonemes_processed.attention_mask.ne(1), -100
        )

        return [audios_processed, phonemes_processed]

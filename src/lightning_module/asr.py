import datetime
import json
import logging
import os
import warnings
from typing import Dict, List, Tuple, Union

import torchmetrics
import hydra

# import lightning as L
import matplotlib.pyplot as plt
import numpy
import pytorch_lightning as L
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

# Disable annoying warnings from Huggingface transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
plt.rcParams["figure.max_open_warning"] = 50

# Set environment variables for Huggingface cache, for datasets and transformers models
# (should be defined before importing datasets and transformers modules)
dir_huggingface_cache_path: str = "/home/Donnees/Data/Huggingface_cache"
os.environ["HF_HOME"] = dir_huggingface_cache_path
os.environ["HF_DATASETS_CACHE"] = dir_huggingface_cache_path + "/datasets"
os.environ["TRANSFORMERS_CACHE"] = dir_huggingface_cache_path + "/models"

# Set environment variables for full trace of errors
os.environ["HYDRA_FULL_ERROR"] = "1"
dir_path: str = str(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
now: str = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S/")

logger: logging.Logger = logging.getLogger(__name__)

epochs = []
logits_to_plot = numpy.zeros((1000, 30))


class NeuralNetwork(L.LightningModule):
    """
    Constructs a neural_network. Inherits from LightningModule.

    Args:
        tokenizer (`PreTrainedTokenizer`): The tokenizer filled with vocab.json needed to know pad_token_id and vocab_size.
        lr (float): The learning rate.
        model (kwargs): Parameters for instantiating the pre-trained model.
        model_class (str): The class of the model one wants to instantiate; expl: 'transformers.Wav2Vec2ForCTC'.
        scheduler (torch.optim.scheduler): Class for instantiating any scheduler one wants; expl: 'OneCycleLR'.
        ckpt_path (str): Path for the checkpoint needed to load the model when testing.
        phonemizer (kwargs): Parameters for instantiating the phonemizer.phonemize class.
        nb_steps_per_epoch (int): Defaults to 1. Number of steps per epoch.
        unfreeze_at_step (int): Defaults to 0. Once per training, the learning strategy freezes all transformer layers at the start of the training and unfreezes them if global_step > unfreeze_at_step.
        is_test (bool): Defaults to False. Determines whether we are training + validating or only testing.
        is_continue_training (bool): Defaults to False. Determines if the training continues from ckpt.
        task_type (str): Defaults to 'phoneme'. Whether to do Speech-to-Text or Speech-to-Phoneme.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        lr: float,
        model,
        model_class: str,
        scheduler: torch.optim.lr_scheduler,
        ckpt_path: str,
        phonemizer,
        nb_steps_per_epoch: int = 1,
        unfreeze_at_step: int = 0,
        is_test: bool = False,
        is_continue_training: bool = False,
        task_type: str = "phoneme",
    ) -> None:
        super(NeuralNetwork, self).__init__()

        self.save_hyperparameters()

        self.lr: float = lr
        self.scheduler = scheduler
        self.unfreeze_at_step: int = unfreeze_at_step
        self.max_steps = nb_steps_per_epoch * self.scheduler.max_epochs

        # Init the processor with the feature extractor and the tokenizer
        self.tokenizer = tokenizer
        self.blank_id: int = self.tokenizer.word_delimiter_token_id
        self.task_type: str = task_type

        # Init the model
        if is_test:
            self.model = hydra.utils.get_method(model_class + ".from_pretrained")(
                dir_path + "/models/" + ckpt_path,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                apply_spec_augment=False,
                vocab_size=len(self.tokenizer),
            )
        elif is_continue_training:
            self.model = hydra.utils.get_method(model_class + ".from_pretrained")(
                dir_path + "/models/" + ckpt_path,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                vocab_size=len(self.tokenizer),
            )
        else:
            self.model = hydra.utils.get_method(model_class + ".from_pretrained")(
                **model,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                vocab_size=len(self.tokenizer),
            )

        logger.info(f"Model {self.model.name_or_path}")

        assert self.task_type != ""
        assert self.task_type is not None
        if self.task_type == "phoneme":
            self.val_per: torchmetrics.text.CharErrorRate = (
                torchmetrics.text.CharErrorRate()
            )
            self.train_per: torchmetrics.text.CharErrorRate = (
                torchmetrics.text.CharErrorRate()
            )
            self.test_per: torchmetrics.text.CharErrorRate = (
                torchmetrics.text.CharErrorRate()
            )
            self.val_per_no_ws: torchmetrics.text.CharErrorRate = (
                torchmetrics.text.CharErrorRate()
            )
            self.test_per_no_ws: torchmetrics.text.CharErrorRate = (
                torchmetrics.text.CharErrorRate()
            )

        elif self.task_type == "text":
            self.val_wer: torchmetrics.text.WordErrorRate = (
                torchmetrics.text.WordErrorRate()
            )
            self.val_cer: torchmetrics.text.CharErrorRate = (
                torchmetrics.text.CharErrorRate()
            )
            self.train_wer: torchmetrics.text.WordErrorRate = (
                torchmetrics.text.WordErrorRate()
            )
            self.test_wer: torchmetrics.text.WordErrorRate = (
                torchmetrics.text.WordErrorRate()
            )
            self.test_cer: torchmetrics.text.CharErrorRate = (
                torchmetrics.text.CharErrorRate()
            )
            self.test_per: torchmetrics.text.CharErrorRate = (
                torchmetrics.text.CharErrorRate()
            )
            self.phonemizer_backend = hydra.utils.instantiate(phonemizer)

            # Freeze the feature encoder part since we won't be training it
        self.model.freeze_feature_extractor()

        logger.info("Model initialization")

    def forward(self, batch: Dataset) -> PreTrainedModel:
        inputs, target = batch["input_values"], batch["labels"]
        return self.model(input_values=inputs, labels=target)

    def on_train_start(self) -> None:
        self.once = False
        if self.unfreeze_at_step > 0:
            logger.info("Entering freeze transformer layers learning strategy")
            for param in self.model.wav2vec2.encoder.layers.parameters():
                param.requires_grad = False
            self.once = True

    def training_step(self, batch: Dataset, batch_idx: int) -> float:
        if self.once and 0 < self.unfreeze_at_step < self.global_step:
            logger.info("Entering unfreeze transformer layers learning strategy")
            for param in self.model.wav2vec2.encoder.layers.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
            self.once = False

        loss = self.forward(batch).loss

        logits, predicted_ids, target = self.compute_metrics(batch)

        if self.task_type == "phoneme":
            self.train_per.update(preds=predicted_ids, target=target)
            train_er_ctc: torch.Tensor = self.train_per.compute()
        elif self.task_type == "text":
            self.train_wer.update(preds=predicted_ids, target=target)
            train_er_ctc: torch.Tensor = self.train_wer.compute()

        self.log("step", int(self.global_step), prog_bar=True, on_step=True)
        self.log("frozen_enc", self.once, prog_bar=True, on_step=True)

        self.plot_posteriogram_every_n_steps(
            val_train="train",
            step=self.global_step,
            n=int(self.trainer.log_every_n_steps * 5),  # logs twice per epoch
            logits=logits,
            gt=target,
            out=predicted_ids,
            er=train_er_ctc.item(),
            log_tensorboard=True,
            blank_id=self.blank_id,
            show_console=False,
        )

        self.log("train/loss_ctc", loss, on_step=True)
        self.log("epoch", self.current_epoch, on_step=True)

        return loss

    def training_epoch_end(self, outputs: float) -> None:
        if self.task_type == "phoneme":
            self.train_per.reset()
        elif self.task_type == "text":
            self.train_wer.reset()

    def validation_step(
        self, batch: Dataset, batch_idx: int
    ) -> Tuple[float, torch.Tensor, str, str]:
        loss = self.forward(batch).loss

        logits, predicted_ids, target = self.compute_metrics(batch)

        if self.task_type == "phoneme":
            self.val_per.update(preds=predicted_ids, target=target)
            self.val_per_no_ws.update(
                preds=predicted_ids.replace(" ", ""),
                target=target.replace(" ", ""),
            )
        elif self.task_type == "text":
            self.val_wer.update(preds=predicted_ids, target=target)
            self.val_cer.update(preds=predicted_ids, target=target)

        self.log("val/loss_ctc", loss, on_epoch=True)

        return loss, logits, predicted_ids, target

    def validation_epoch_end(
        self, outputs: Union[torch.Tensor, Tuple[float, torch.Tensor, str, str]]
    ) -> None:
        logits = torch.nn.Softmax(dim=2)(outputs[0][1])
        predicted_ids = outputs[0][2]
        target = outputs[0][3]
        decoded_text, ground_truth_text = [], []
        for i in range(len(outputs)):
            decoded_text.append(outputs[i][2])
            ground_truth_text.append(outputs[i][3])

        if self.task_type == "phoneme":
            er_ctc: torch.Tensor = self.val_per.compute()
            self.log("val/per_ctc", er_ctc, on_epoch=True)
            self.log("val/per_ctc_no_ws", self.val_per_no_ws.compute(), on_epoch=True)
        elif self.task_type == "text":
            er_ctc: torch.Tensor = self.val_wer.compute()
            self.log("val/wer_ctc", er_ctc, on_epoch=True)
            self.log("val/cer_ctc", self.val_cer.compute(), on_epoch=True)

        self.plot_posteriogram_every_n_steps(
            val_train="validation",
            step=self.current_epoch,
            n=1,  # once per epoch
            logits=logits,  # same logit for same predicted id for same target
            gt=target,
            out=predicted_ids,
            er=er_ctc.item(),
            log_tensorboard=True,
            blank_id=self.blank_id,
            show_console=False,
        )

        self.log_decoded_text(
            self.current_epoch, decoded_text[0:15], ground_truth_text[0:15]
        )

        if self.task_type == "phoneme":
            self.val_per.reset()
        elif self.task_type == "text":
            self.val_wer.reset()
            self.val_cer.reset()

    def test_step(self, batch: Dataset, batch_idx: int) -> Tuple[str, str]:
        loss = self.forward(batch).loss

        logits, predicted_ids, target = self.compute_metrics(batch)

        if self.task_type == "phoneme":
            self.test_per.update(preds=predicted_ids, target=target)
            self.test_per_no_ws.update(
                preds=predicted_ids.replace(" ", ""),
                target=target.replace(" ", ""),
            )
        elif self.task_type == "text":
            self.test_wer.update(preds=predicted_ids, target=target)
            self.test_cer.update(preds=predicted_ids, target=target)

        self.log("~test/loss_ctc", loss, on_epoch=True)

        return predicted_ids, target

    def test_epoch_end(self, outputs: List[Tuple[str, str]]) -> None:
        decoded_text, ground_truth_text = [], []
        for i in range(len(outputs)):
            decoded_text.append(outputs[i][0])
            ground_truth_text.append(outputs[i][1])

        if self.task_type == "phoneme":
            self.log("~test/per_ctc", self.test_per.compute(), on_epoch=True)
            self.log(
                "~test/per_ctc_no_ws", self.test_per_no_ws.compute(), on_epoch=True
            )
        elif self.task_type == "text":
            self.compute_per(
                decoded_text=decoded_text, ground_truth_text=ground_truth_text
            )
            self.log("~test/wer_ctc", self.test_wer.compute(), on_epoch=True)
            self.log("~test/cer_ctc", self.test_cer.compute(), on_epoch=True)

        self.log_decoded_text(
            self.current_epoch, decoded_text[0:15], ground_truth_text[0:15]
        )

    def compute_per(  # useful only for latin alphabet texts
        self, decoded_text: List[str], ground_truth_text: List[str]
    ) -> None:
        """
        Calls the phonemizer backend to phonemize and compute the PER metric.

        Args:
            decoded_text (List[str]): List of OUT texts from test_step().
            ground_truth_text (List[str]): List of GT texts from test_step().
        Returns:
            NoneType: None
        """
        self.test_per.update(
            preds=self.phonemizer_backend.phonemize(decoded_text),
            target=self.phonemizer_backend.phonemize(ground_truth_text),
        )
        self.log("~test/per_ctc", self.test_per.compute(), on_epoch=True)

    def configure_optimizers(self):
        optimizer: torch.optim.Adam = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
        scheduler = {
            "scheduler": hydra.utils.instantiate(
                self.scheduler.trapeze,
                optimizer=optimizer,
                warmup_steps=int(1e-1 * self.max_steps),
                hold_steps=int(4e-1 * self.max_steps),
                decay_steps=int(5e-1 * self.max_steps),
                total_steps=self.max_steps,
            ),
            # "scheduler": hydra.utils.instantiate(
            #     self.scheduler,
            #     optimizer=optimizer,
            #     max_lr=self.lr,
            #     steps_per_epoch=self.trainer.log_every_n_steps * 10,
            # ),
            "interval": "step",
        }
        logger.info(f"optimizer = {optimizer}, scheduler = {scheduler}")
        return [optimizer], [scheduler]

    def compute_metrics(self, batch: Dataset) -> List[Union[str, torch.Tensor]]:
        """
        Does forward, calculates logits, decodes logits.

        Args:
            batch (Dataset): the train or validation or test batch.
        Returns:
            List[Union[str, torch.Tensor]]: logits, predicted_ids, target
        """
        logits = self.forward(batch).logits
        inputs, target = batch["input_values"], batch["labels"]
        predicted_ids = torch.argmax(logits, dim=2)

        predicted_ids = torch.flatten(predicted_ids)
        target = torch.flatten(target)

        target[target == -100] = self.tokenizer.pad_token_id

        predicted_ids = self.tokenizer.decode(predicted_ids)
        target = self.tokenizer.decode(target, group_tokens=False)

        return [logits, predicted_ids.strip(), target.strip()]

    def plot_posteriogram_every_n_steps(
        self,
        val_train: str,
        step: int,
        n: int,
        logits: torch.Tensor,
        gt: str,
        out: str,
        er: float,
        log_tensorboard: bool,
        blank_id: int,
        show_console: bool = False,
    ) -> None:
        """
        Logs the posteriogram in console and/or in tensorboard every n epochs.

        Args:
            val_train (str): The posteriogram is for validation or training.
            step (int): step or epoch.
            n (int): Logs every n `step`.
            logits (`torch.Tensor`): The logits from the prediction.
            gt (str): The ground truth string.
            out (str): The predicted string by the model.
            er (float): The ER at `step`.
            log_tensorboard (bool): Boolean to log in tensorboard.
            blank_id (int): The id for the word_delimiter_token.
            show_console (bool): Boolean to display in console.
        Returns:
            NoneType: None
        """

        s_length = numpy.linspace(1, len(logits[0]), len(logits[0]))
        if step % n == 0:  # plot every n steps
            fig, ax = plt.subplots(
                figsize=(18, 4), dpi=100
            )  # res = (x_axis*dpi, y_axis*dpi)
            for i in range(len(logits[0][0])):  # range of vocab.size
                blank_token = torch.argmax(logits[0, :, i], dim=0)
                if blank_token.item() == blank_id:
                    ax.plot(
                        s_length,
                        torch.log(logits[0, :, i]).cpu().detach().numpy(),
                        "ro-",
                    )  # logarithmic scale
                else:
                    ax.plot(
                        s_length, torch.log(logits[0, :, i]).cpu().detach().numpy()
                    )  # logarithmic scale

            ax.set_title(f"GT : {gt}\nOUT : {out}", loc="center", wrap=True)
            ax.set_xlabel(
                f"Time    epoch {'P' if self.task_type == 'phoneme' else 'W' if self.task_type == 'text' else Exception(f'task_type must be `phoneme` or `text` but it was {self.task_type}')}ER = {er * 100:.1f}%"
            )
            ax.set_ylabel("Logits (log scale)")
            fig.tight_layout()

            if show_console:
                plt.show()

            if log_tensorboard:
                if val_train == "train":
                    str_value = "step"
                else:
                    str_value = "epoch"
                self.logger.experiment.add_figure(
                    f"{val_train}/posteriogram-{str_value}{step}",
                    figure=fig,
                    global_step=step,
                )
            plt.close()

    def log_decoded_text(
        self, step: int, predicted_ids: List[str], target: List[str]
    ) -> None:
        """
        Logs decoded audio versus sentence.

        Args:
            step (int): log_decoded_text logs every step.
            predicted_ids (List[str]): The OUT strings.
            target (List[str]): The Ground Truth strings.
        Returns:
            NoneType: None
        """
        file_to_log = {}
        for i in range(len(predicted_ids)):
            file_to_log[f"OUT {i}"] = predicted_ids[i]
            file_to_log[f"GT {i}"] = target[i]

        self.logger.experiment.add_text(
            "comparison", pretty_json(file_to_log), global_step=step
        )


def pretty_json(hp: Dict[str, str]) -> str:
    """
    Pretties a Dict instance in raw json-ed text.

    Args:
        hp (Dict[str, str]): The Dict instance.
    Returns:
        str: hp
    """
    json_hp = json.dumps(hp, indent=2, ensure_ascii=False)
    return "".join("\t" + line for line in json_hp.splitlines(True))

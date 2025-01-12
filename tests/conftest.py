import pytest
import hydra
import torch
import transformers

from vibravox.lightning_datamodules.spkv import SPKVLightningDataModule
from vibravox.lightning_datamodules.stp import STPLightningDataModule
from vibravox.lightning_datamodules.bwe import BWELightningDataModule
from vibravox.torch_modules.dnn.eben_generator import EBENGenerator
from vibravox.torch_modules.losses.feature_loss import (
    FeatureLossForDiscriminatorMelganMultiScales,
)
from vibravox.torch_modules.losses.hinge_loss import (
    HingeLossForDiscriminatorMelganMultiScales,
)
from vibravox.torch_modules.dnn.melgan_discriminator import (
    MelganMultiScalesDiscriminator,
)


@pytest.fixture(params=[16000])
def sample_rate(request) -> int:
    """Sample frequency."""
    return request.param


@pytest.fixture(params=[4])
def batch_size(request) -> int:
    """Number of files."""
    return request.param


@pytest.fixture(params=[15679])
def time_len(request) -> int:
    """Number of time samples."""
    return request.param


@pytest.fixture
def sample(batch_size, time_len) -> torch.Tensor:
    """
    Produce a batch of samples with size (batch_size, 1, time_len)
    """
    return torch.randn(batch_size, 1, time_len)


@pytest.fixture
def bwe_lightning_datamodule_instance_from_hydra() -> BWELightningDataModule:
    with hydra.initialize(
        version_base="1.3", config_path="../configs"
    ):
        overrides = [f"lightning_datamodule='bwe'"]
        cfg = hydra.compose(config_name="run", overrides=overrides)

        return hydra.utils.instantiate(cfg.lightning_datamodule)


@pytest.fixture
def stp_lightning_datamodule_instance_from_hydra() -> STPLightningDataModule:
    with hydra.initialize(
        version_base="1.3", config_path="../configs"
    ):
        overrides = [f"lightning_datamodule='stp'"]
        cfg = hydra.compose(config_name="run", overrides=overrides)

        return hydra.utils.instantiate(cfg.lightning_datamodule)


@pytest.fixture
def spkv_lightning_datamodule_instance_from_hydra() -> SPKVLightningDataModule:
    with hydra.initialize(
        version_base="1.3", config_path="../configs"
    ):
        overrides = [f"lightning_datamodule='spkv'"]
        cfg = hydra.compose(config_name="run", overrides=overrides)

        return hydra.utils.instantiate(cfg.lightning_datamodule)

@pytest.fixture(
    params=["temple_vibration_pickup",
            "throat_microphone",
            "rigid_in_ear_microphone",
            "soft_in_ear_microphone",
            "forehead_accelerometer",
            "headset_microphone"]
)
def sensor_name(request) -> str:
    return request.param


@pytest.fixture(
    params=["Cnam-LMSSC/vibravox-test"]  # "Cnam-LMSSC/vibravox_enhanced_by_EBEN"
    # To avoid downloading too much files we only test the smaller dataset
)
def dataset_name(request) -> str:
    return request.param

@pytest.fixture(
    params=["speech_noisy"]  # "speech_clean"
    # To avoid downloading too much files wee only test the smaller subset
)
def subset_name(request) -> str:
    return request.param


@pytest.fixture(params=[True, False])
def streaming(request) -> bool:
    return request.param


@pytest.fixture(params=["pad", "constant_length-3000-ms"])
def collate_strategy(request) -> str:
    return request.param


@pytest.fixture
def bwe_lightning_datamodule_instance(
    sample_rate, dataset_name, subset_name, sensor_name, collate_strategy, streaming, batch_size
) -> BWELightningDataModule:
    """BWELightningDataModule instance."""

    if sensor_name == "headset_microphone":
        pytest.skip("Skipping for headset_microphone")

    datamodule = BWELightningDataModule(
        sample_rate=sample_rate,
        dataset_name_principal=dataset_name,
        subset=subset_name,
        sensor=sensor_name,
        collate_strategy=collate_strategy,
        streaming=streaming,
        batch_size=batch_size,
    )

    return datamodule


@pytest.fixture
def stp_lightning_datamodule_instance(
    sample_rate, dataset_name, subset_name, sensor_name, streaming, batch_size
) -> STPLightningDataModule:
    """STPLightningDataModule instance."""

    feature_extractor = transformers.Wav2Vec2FeatureExtractor()
    tokenizer = transformers.Wav2Vec2CTCTokenizer.from_pretrained("Cnam-LMSSC/vibravox-phonemes-tokenizer")

    datamodule = STPLightningDataModule(
        sample_rate=sample_rate,
        dataset_name=dataset_name,
        subset=subset_name,
        sensor=sensor_name,
        streaming=streaming,
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    return datamodule

@pytest.fixture
def spkv_lightning_datamodule_same_sensors_instance(
    sample_rate, dataset_name, subset_name, sensor_name,
) -> SPKVLightningDataModule:
    """SPKVLightningDataModule instance."""

    datamodule = SPKVLightningDataModule(
        sample_rate=sample_rate,
        dataset_name=dataset_name,
        subset=subset_name,
        sensor_a=sensor_name,
        sensor_b=sensor_name,
        pairs="mixed_gender",
        streaming=False,
        batch_size=1,
    )

    return datamodule


@pytest.fixture
def discriminator_melgan_multiscales_instance(
    sample_rate,
) -> MelganMultiScalesDiscriminator:
    """DiscriminatorMelganMultiScales model instance."""
    # Disabling gradient computation in each test
    torch.set_grad_enabled(False)
    discriminator_melgan_multiscales = MelganMultiScalesDiscriminator(
        sample_rate=sample_rate
    )
    discriminator_melgan_multiscales = discriminator_melgan_multiscales.to("cpu")
    discriminator_melgan_multiscales.eval()
    return discriminator_melgan_multiscales


@pytest.fixture
def eben_generator_instance() -> EBENGenerator:
    """Produce EBENGenerator instance."""
    eben_generator = EBENGenerator(m=4, n=32, p=1)
    return eben_generator


@pytest.fixture
def feature_loss_for_discriminator_melgan_multi_scales_instance() -> (
    FeatureLossForDiscriminatorMelganMultiScales
):
    """
    Produce DiscriminatorMelganMultiScales instance.
    """
    return FeatureLossForDiscriminatorMelganMultiScales()


@pytest.fixture
def hinge_loss_for_discriminator_melgan_multi_scales_instance() -> (
    HingeLossForDiscriminatorMelganMultiScales
):
    """
    Produce DiscriminatorMelganMultiScales instance.
    """
    return HingeLossForDiscriminatorMelganMultiScales()

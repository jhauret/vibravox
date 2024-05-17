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
        version_base="1.3", config_path="../configs/lightning_datamodule"
    ):
        cfg = hydra.compose(config_name="bwe")
        return hydra.utils.instantiate(cfg)


@pytest.fixture
def stp_lightning_datamodule_instance_from_hydra() -> STPLightningDataModule:
    with hydra.initialize(
        version_base="1.3", config_path="../configs/lightning_datamodule"
    ):
        cfg = hydra.compose(config_name="stp")
        return hydra.utils.instantiate(cfg)


@pytest.fixture(
    params=["body_conducted.temple.contact_microphone",
            "body_conducted.throat.piezoelectric_sensor",
            "body_conducted.in_ear.rigid_earpiece_microphone",
            "body_conducted.in_ear.comply_foam_microphone",
            "body_conducted.forehead.miniature_accelerometer",
            "airborne.mouth_headworn.reference_microphone"]
)
def sensor_name(request) -> str:
    return request.param


@pytest.fixture(
    params=["Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp"]  # "Cnam-LMSSC/vibravox"
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

    if sensor_name == "airborne.mouth_headworn.reference_microphone":
        pytest.skip("Skipping for airborne.mouth_headworn.reference_microphone")

    datamodule = BWELightningDataModule(
        sample_rate=sample_rate,
        dataset_name=dataset_name,
        subset=subset_name,
        sensor=sensor_name,
        collate_strategy=collate_strategy,
        streaming=streaming,
        batch_size=batch_size,
    )
    datamodule.setup(stage="test")

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

    datamodule.setup(stage="test")

    return datamodule


def spkv_lightning_datamodule_instance(
    sample_rate, dataset_name, subset_name, sensor_name,
) -> SPKVLightningDataModule:
    """SPKVLightningDataModule instance."""

    datamodule = SPKVLightningDataModule(
        pklfile_path="configs/lightning_datamodule/spkv_pairs/pairs.pkl",
        sample_rate=sample_rate,
        dataset_name=dataset_name,
        subset=subset_name,
        sensor_a=sensor_name,
        #keep sensor_b to default value which is "airborne.mouth_headworn.reference_microphone"
        streaming=False,
        batch_size=1,
    )

    datamodule.setup(stage="test")

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

import pytest
import hydra
import torch

from vibravox.lightning_datamodules.stp import STPLightningDataModule
from vibravox.lightning_datamodules.bwe import BWELightningDataModule
from vibravox.torch_modules.dnn.eben_generator import EBENGenerator
from vibravox.torch_modules.losses.feature_loss import (
    FeatureLossForDiscriminatorMelganMultiScales,
)
from vibravox.torch_modules.losses.hinge_loss import HingeLossForDiscriminatorMelganMultiScales
from vibravox.torch_modules.dnn.melgan_discriminator import MelganMultiScalesDiscriminator


@pytest.fixture(params=[16000])
def sample_rate(request) -> int:
    """Sample frequency."""
    return request.param


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    """Number of files."""
    return request.param


@pytest.fixture(params=[15999, 16000, 16001])
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
    params=["bwe_in-ear_rigid_earpiece_microphone"]
)  # , "bwe_throat_piezoelectric_sensor"
def bwe_subset_name(request) -> str:
    return request.param

@pytest.fixture(
    params=["asr_in-ear_rigid_earpiece_microphone"]
)  # , "bwe_throat_piezoelectric_sensor"
def asr_subset_name(request) -> str:
    return request.param



@pytest.fixture(params=[True, False])
def streaming(request) -> bool:
    return request.param


@pytest.fixture
def bwe_lightning_datamodule_instance(
    sample_rate, bwe_subset_name, streaming, batch_size
) -> BWELightningDataModule:
    """BWELightningDataModule instance."""

    datamodule = BWELightningDataModule(sample_rate, bwe_subset_name, streaming, batch_size)
    datamodule.setup()

    return datamodule


@pytest.fixture
def stp_lightning_datamodule_instance(
    sample_rate, asr_subset_name, streaming, batch_size
) -> STPLightningDataModule:
    """STPLightningDataModule instance."""

    datamodule = STPLightningDataModule(sample_rate, asr_subset_name, streaming, batch_size)
    datamodule.setup()

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

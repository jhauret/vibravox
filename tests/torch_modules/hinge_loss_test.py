import torch


class TestHingeLossForDiscriminatorMelganMultiScales:
    def test_forward_fake(
        self,
        sample,
        discriminator_melgan_multiscales_instance,
        hinge_loss_for_discriminator_melgan_multi_scales_instance,
    ):
        """Test the hinge loss for fake sample target."""
        embeddings = discriminator_melgan_multiscales_instance(sample)
        loss = hinge_loss_for_discriminator_melgan_multi_scales_instance(
            embeddings, target=-1
        )

        # Make sure loss is a scalar
        assert loss.shape == torch.Size([])

    def test_forward_real(
        self,
        sample,
        discriminator_melgan_multiscales_instance,
        hinge_loss_for_discriminator_melgan_multi_scales_instance,
    ):
        """Test the hinge loss for fake sample target."""
        embeddings = discriminator_melgan_multiscales_instance(sample)
        loss = hinge_loss_for_discriminator_melgan_multi_scales_instance(
            embeddings, target=1
        )

        # Make sure loss is a scalar
        assert loss.shape == torch.Size([])

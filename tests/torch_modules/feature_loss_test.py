import torch


class TestHingeLossForDiscriminatorMelganMultiScales:
    def test_forward(
        self,
        sample,
        discriminator_melgan_multiscales_instance,
        feature_loss_for_discriminator_melgan_multi_scales_instance,
    ):
        """Test the feature loss forward"""
        embeddings_a = discriminator_melgan_multiscales_instance(sample)
        embeddings_b = discriminator_melgan_multiscales_instance(sample)
        loss = feature_loss_for_discriminator_melgan_multi_scales_instance(
            embeddings_a, embeddings_b
        )

        # Make sure loss is a scalar
        assert loss.shape == torch.Size([])

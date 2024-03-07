import torch

class TestDiscriminatorMelganMultiScales:
    def test_forward_output_format(self, sample, discriminator_melgan_multiscales_instance):
        """Check that the output shape is correct"""
        scales_embeddings = discriminator_melgan_multiscales_instance(sample)

        assert isinstance(scales_embeddings, list)
        assert len(scales_embeddings) == len(
            discriminator_melgan_multiscales_instance.discriminators
        )
        assert all([isinstance(x[-1], torch.Tensor) for x in scales_embeddings])

    def test_minimum_number_of_parameters(self, discriminator_melgan_multiscales_instance):
        """Check that the number of parameters is greater than 1e3"""
        total_params = sum(
            p.numel() for p in discriminator_melgan_multiscales_instance.parameters()
        )

        assert total_params > 1e3

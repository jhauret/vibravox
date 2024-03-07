class TestEBENGenerator:
    def test_forward_output_format(self, sample, eben_generator_instance):
        corrupted_signal = eben_generator_instance.cut_to_valid_length(sample)
        enhanced_signal, enhanced_signal_decomposed = eben_generator_instance(corrupted_signal)

        assert enhanced_signal.shape == corrupted_signal.shape

    def test_minimum_number_of_parameters(self, eben_generator_instance):
        """Check that the number of parameters is greater than 1e3"""
        total_params = sum(
            p.numel() for p in eben_generator_instance.parameters()
        )

        assert total_params > 1e3
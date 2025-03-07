import torch


class TestCausalConv1d:
    def test_forward_streaming_vs_non_streaming(
        self,
        sample,
        causal_conv1d_instance
    ):
        # ----- Non-streaming forward -----
        causal_conv1d_instance.set_streaming(False)
        out_non_stream = causal_conv1d_instance(sample)

        # ----- Streaming forward -----
        causal_conv1d_instance.set_streaming(True)
        causal_conv1d_instance.reset_state()

        chunk_size = 1  # Example chunk size
        streaming_outputs = []
        for start_idx in range(0, sample.shape[-1], chunk_size):
            end_idx = start_idx + chunk_size
            chunk = sample[:, :, start_idx:end_idx]
            out_chunk = causal_conv1d_instance(chunk)
            streaming_outputs.append(out_chunk)
        out_stream = torch.cat(streaming_outputs, dim=-1)
    
        # 1. Compare shapes
        assert out_non_stream.shape == out_stream.shape, (
            f"Shape mismatch: non-stream={out_non_stream.shape}, "
            f"streaming={out_stream.shape}"
        )

        # 2. Compare values directly (within some small tolerance)
        #    If you want exact match, you can remove the tolerances,
        #    but floating-point ops often need a small tolerance.
        assert torch.allclose(out_non_stream, out_stream, rtol=1e-5, atol=1e-8), (
            "Non-streaming and streaming outputs differ beyond tolerances."
        )

        # (Optional) If you still want to quantify the difference:
        # difference = (out_non_stream - out_stream).abs().max().item()
        # assert difference < 1e-7, f"Max difference is too large: {difference}"
"""CPU checks of the codec; does not load a diffusion model or use the GPU."""
import numpy as np
from ecc_utils import LDPCCode


def main():
    rng = np.random.RandomState(2024)
    for n in (98, 623):
        code = LDPCCode.from_rate(n, 2.0 / 3.0, seed=20260826, column_weight=3)
        secret = rng.randint(0, 2, size=code.k).astype(np.int8)
        coded = code.encode(secret)
        assert code.check(coded)
        decoded = code.decode(coded, (1.0 - 2.0 * coded) * 8.0, max_iterations=100)
        assert np.array_equal(decoded['secret_bits'], secret)
        print('LDPC noiseless round trip passed: n={}, k={}'.format(n, code.k))
    try:
        import torch
    except ImportError:
        print('SKIPPED: DCT/QIM check requires torch.')
        return
    from qim_utils import dct2, idct2, embed_message_in_latents, extract_message_details_from_latents
    torch.manual_seed(1500)
    latent = torch.randn(1, 4, 64, 64)
    assert torch.allclose(idct2(dct2(latent)), latent, atol=1e-4)
    stego = embed_message_in_latents(latent, coded, key_seed=999999,
        step=3.0, repeat=5, min_radius=8.0, max_radius=32.0)
    extracted = extract_message_details_from_latents(stego, n_bits=623,
        key_seed=999999, step=3.0, repeat=5, min_radius=8.0, max_radius=32.0)
    assert np.array_equal(extracted['soft_bits'], coded)
    decoded = code.decode(extracted['soft_bits'], extracted['message_llrs'])
    assert np.array_equal(decoded['secret_bits'], secret)
    print('DCT round trip and QIM + LDPC noiseless extraction passed.')


if __name__ == '__main__':
    main()

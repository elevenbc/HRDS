# Third-party notices

- `inverse_stable_diffusion.py`, `modified_stable_diffusion.py` and `optim_utils.py` were obtained from the working Tree-Ring project. Preserve the upstream MIT notice in `licenses/Tree-Ring-MIT.txt` (Copyright 2023 Yuxin Wen). The inversion source also credits efficient-prompt-to-prompt. Upstream: https://github.com/YuxinWenRick/tree-ring-watermark and https://github.com/cccntu/efficient-prompt-to-prompt . Local modifications are included as present in the experimental workspace.
- `calculate_fid_score.py` carries the JKU Linz Apache-2.0 header and attributes TTUR. `inception.py` implements the PyTorch FID Inception network, corresponding to pytorch-fid. Preserve the Apache-2.0 text in `licenses/Apache-2.0.txt` and all source headers. Upstream: https://github.com/bioinf-jku/TTUR and https://github.com/mseitzer/pytorch-fid . The Inception loader in this workspace reads a local checkpoint.
- PyTorch, Diffusers, Transformers, LPIPS, scikit-image and other installed packages retain their respective licenses; their source and model weights are not bundled here.

The upstream notice does not establish a blanket license for this project's original additions. No new grant of rights is made on behalf of the authors during packaging.

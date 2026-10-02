# NASWOT Baseline (PyTorch 2.0.1, NAS-Bench-201)

A working, reproducible baseline of **Neural Architecture Search Without Training** (Mellor et al., 2021), adapted to run on PyTorch 2.x with CUDA 11.8 on Windows.

Based on the original repository: https://github.com/BayesWatch/nas-without-training

## What changed from upstream

- Environment rebuilt without `conda-forge` (mixing it with the `pytorch`/`nvidia` channels caused a `WinError 182` DLL load failure on `nvfuser_codegen.dll`).
- In-place ops are disabled before hooks are registered (`module.inplace = False`), which avoids the "view is being modified inplace" autograd error.
- Backward hooks use `register_full_backward_hook` (PyTorch 2.x compatible).
- Scope is **NAS-Bench-201 only** (NAS-Bench-101 needs TensorFlow and is not supported here).

## Setup

```bash
conda config --set channel_priority strict
conda env create -f environment.yml --solver=libmamba
conda activate naswot
```

`environment.yml` uses the `pytorch`, `nvidia` and `defaults` channels only. Do **not** add `conda-forge`.

### Benchmark file

Download the NAS-Bench-201 file (`NAS-Bench-201-v1_0-e61699.pth`, ~2 GB) from the [NAS-Bench-201 repository](https://github.com/D-X-Y/NAS-Bench-201) and place it in the location expected by `score_networks.py` (check the `--api_loc` argument, or the default path in the script). It is not committed to this repo.

CIFAR-10 is downloaded automatically by torchvision on first run.

## Verify the installation

**1. Check PyTorch and CUDA**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.0.1 True <your GPU name>`. If this raises the `nvfuser_codegen.dll` error, the env has a conda channel clash; rebuild it without `conda-forge`.

**2. Check the NAS-Bench-201 API import**

```bash
python -c "from nas_201_api import NASBench201API as API; print('API OK')"
```

**3. Run the baseline scoring**

```bash
python .\score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --dataset cifar10
```

If setup is correct, the script prints the GPU in use (for example `Using GPU 0: NVIDIA GeForce GTX 1650 (PyTorch CUDA 11.8)`) and then starts printing `hook_logdet` scores for the sampled architectures, with no errors. For a quick smoke test, interrupt it after a few scores print.

## Troubleshooting

| Symptom                                                                            | Cause                                                | Fix                                                                                                                            |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `OSError: [WinError 182] ... nvfuser_codegen.dll`                                  | Mixed conda channels give mismatched MKL/OpenMP DLLs | Rebuild env without `conda-forge`, or install torch from the pip wheels (`--index-url https://download.pytorch.org/whl/cu118`) |
| `Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace` | In-place ReLU after a full backward hook             | Set `inplace = False` on all modules before registering hooks (already done in this repo)                                      |
| `conda env create` hangs                                                           | Old conda solver                                     | Use `--solver=libmamba` (conda 22.11+)                                                                                         |
| `numpy` import errors with torch 2.0.1                                             | NumPy 2.x is incompatible                            | Keep `numpy<2`                                                                                                                 |

## Credits and license

This work builds on _Neural Architecture Search Without Training_ by Joseph Mellor, Jack Turner, Amos Storkey and Elliot J. Crowley (ICML 2021). The original license is retained in `LICENSE`.

```bibtex
@inproceedings{mellor2021neural,
  title={Neural Architecture Search without Training},
  author={Mellor, Joseph and Turner, Jack and Storkey, Amos and Crowley, Elliot J},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```

# CCGL: Contrastive Cascade Graph Learning

PyTorch implementation of [CCGL](https://doi.org/10.1109/TKDE.2022.3151829)
(TKDE 2023): cascade augmentation, contrastive pre-training, fine-tuning and
self-distillation. Old TensorFlow code is archived in [legacy/tensorflow](legacy/tensorflow).

## Setup

Use Python 3.12 and a CUDA build of PyTorch suitable for your GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the data from
[Google Drive](https://drive.google.com/file/d/1wmUa7hvJlF5oCLVJ72OgyKnVkHZJX8jX/view),
[OneDrive](https://1drv.ms/u/s!AsVLooK4NjBruTngZWgx1p0psD1k?e=5iMcVB) or
[Baidu Netdisk](https://pan.baidu.com/s/16A5SvCVEtPYxelyYIlYAQg) (`trqg`).
Place each raw file at `datasets/<dataset>/dataset.txt`.

## Run

```bash
python -m ccgl.prepare --dataset weibo \
    --raw datasets/weibo/dataset.txt --output prepared/weibo.pt
python -m ccgl.train --data prepared/weibo.pt --output results/main/weibo
```

Training defaults to one run of all three label fractions: 1%, 10% and 100%.
Use `--runs N` for independent repetitions. To run all five datasets five times each:

```bash
bash run.sh 5
```

Prepared data and completed fractions are reused. Unfinished work restarts from
its seed; use a fresh output directory when changing training settings.

## Outputs

Checkpoints, test predictions, metrics and logs are saved under
`results/main/<dataset>/run-<seed>/`.

See [implementation differences](IMPLEMENTATION-DIFFERENCES.md) for the changes
from the original code.

## Citation

```bibtex
@article{xu2023ccgl,
  author = {Xovee Xu and Fan Zhou and Kunpeng Zhang and Siyuan Liu},
  title = {{CCGL}: Contrastive Cascade Graph Learning},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume = {35},
  number = {5},
  pages = {4539--4554},
  year = {2023},
  doi = {10.1109/TKDE.2022.3151829}
}
```

## License

Code is released under the [MIT License](LICENSE).

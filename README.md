# CCGL: Contrastive Cascade Graph Learning

PyTorch implementation of [CCGL](https://doi.org/10.1109/TKDE.2022.3151829)
(TKDE 2023): cascade augmentation, contrastive pre-training, fine-tuning and
self-distillation. Old TensorFlow code is archived in [legacy/tensorflow](legacy/tensorflow).

## Setup

Use Python 3.12 and a PyTorch build suitable for your hardware.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download [cascades.zip](https://drive.google.com/file/d/1r-OHiiO9QdYH8sBc6tjmOd-vb_rJCYBo/view?usp=sharing) and extract it into `datasets/`.
See the [data format](datasets/README.md).

## Run

```bash
python -m ccgl.prepare --dataset weibo \
    --raw datasets/cascades/weibo.parquet --output prepared/weibo.pt
python -m ccgl.train --data prepared/weibo.pt --output results/main/weibo
```

Training defaults to one run of all three label fractions: 1%, 10% and 100%.
Use `--runs N` for independent repetitions. To run all five datasets five times each:

```bash
bash run.sh 5
```

Prepared data and completed fractions are reused. Unfinished work restarts from
its seed; use a fresh output directory when changing training settings.

## Test

Training evaluates the test split automatically. To evaluate a saved student or teacher:

```bash
python -m ccgl.test --data prepared/weibo.pt \
    --checkpoint results/main/weibo/run-0/100/student.pt \
    --output results/main/weibo/run-0/100/test
```

Use the checkpoint from the desired label fraction (1, 10 or 100).

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

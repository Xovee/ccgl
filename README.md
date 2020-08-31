# CCGL: Contrastive Cascade Graph Learning

[![Open In Colab](./.assets/colab-badge.svg)](https://colab.research.google.com/drive/1-ZXVIWdEvN8rDSa2i5OrV9Ov5nw63we9?usp=sharing)
![](https://img.shields.io/badge/python-3.7-green)
![](https://img.shields.io/badge/tensorflow-2.3-green)
![](https://img.shields.io/badge/cudatoolkit-10.1-green)
![](https://img.shields.io/badge/cudnn-7.6.5-green)

This repo provides a reference implementation of Contrastive Cascade Graph Learning (**CCGL**) framework as described in the paper:

> Graph Data Augmentation for Contrastive Cascade Learning  
> Xovee Xu, Fan Zhou, Kunpeng Zhang, and Goce Trajcevski  
> Submitted for review  




## Dataset

You can download all five datasets (Weibo, Twitter, ACM, APS, and DBLP) via either one of the following links:
- Google Drive: [Download link](https://drive.google.com/drive/folders/1VH8QNdIdF0Jy79gIPGj4eFoZ530r7V8u?usp=sharing)
- OneDrive: [Download link](https://1drv.ms/u/s!AsVLooK4NjBruS6rfLyDRufL3SF1?e=oPzFcZ)
- Tencent Drive: [Download link](https://share.weiyun.com/DG6Tjs5h)
- Baidu Netdisk: [Download link](https://pan.baidu.com/s/1phTUCPgtVpS9aiu1sH6nqA), download password: `ngjt`

## Environmental Settings

Our experiments are conducted on Ubuntu 20.04, a single NVIDIA 1080Ti GPU, 48GB RAM, and Intel i7 8700K. CGC is implemented by `Python 3.7`, `TensorFlow 2.3`, `Cuda 10.1`, and `Cudnn 7.6.5`.

Step 1: Create a virtual environment and install GPU-support packages via [Anaconda](https://www.anaconda.com/):
```shell
conda create --name=ccgl python=3.7 cudatoolkit=10.1 cudnn=7.6.5

conda activate ccgl
```

Step 2: Install all dependencies:
```python
pip install -r requirements.txt
```

## Usage

Here we take Weibo dataset as an example to demonstrate the usage.

### Preprocess

Step 1: divide, filter, generate labeled and unlabeled cascades:
```shell
cd ccgl
# labeled cascades
python src/gene_cas.py --input=./datasets/weibo/ --unlabel=False
# unlabeled cascades
python src/gene_cas.py --input=./datasets/weibo/ --unlabel=True
```

Step 2: augment both labeled and unlabeled cascades (here we use the `AugSIM` strategy):
```shell
python src/augmentor.py --input=./datasets/weibo/ --aug_strategy=AugSIM
```

Step 3: generate cascade embeddings:
```shell
python src/gene_emb.py --input=./datasets/weibo/ 
```

### Pre-training

```shell
python src/pre_training.py --name=weibo-0 --input=./datasets/weibo/ --projection_head=4-1
```

### Fine-tuning

```shell
python src/fine_tuning.py --name=weibo-0 --num=0 --input=./datasets/weibo/ --projection_head=4-1
```

### Distilling

```shell
python src/ditilling.py --nmae=weibo-0-0 --num=0 --input=./datasets/weibo/ --projection_head=4-1
```


### (Optional) Run the Base model

```shell
python src/base_model.py --input=./datasets/weibo/ 
```

## CGC model weights

We provide pre-trained, fine-tuned, and distilled CGC model weights. Please see details in the following table. 

Model|Dataset|Label Fraction|Projection Head|MSLE|Weights
:---|:---|:---|:---|:---|:---
Pre-trained CCGL model|Weibo|100%|4-1|-|[Download](./results/pre_training_weight/weibo-100.h5)
Pre-trained CCGL model|Weibo|10%|4-4|-|[Download](./results/pre_training_weight/weibo-10.h5)
Pre-trained CCGL model|Weibo|1%|4-3|-|[Download](./results/pre_training_weight/weibo-1.h5)
Fine-tuned CCGL model|Weibo|100%|4-1|2.70|[Download](./results/fine_tuning_weight/weibo-100-0.h5)
Fine-tuned CCGL model|Weibo|10%|4-4|2.87|[Download](./results/fine_tuning_weight/weibo-10-0.h5)
Fine-tuned CCGL model|Weibo|1%|4-3|3.30|[Download](./results/fine_tuning_weight/weibo-1-0.h5)

Load weights into the model:
```python
# contruct model, carefully check projection head designs:
# use different number of Dense layers
...
# load weights for fine-tuning, distillation, or evaluation
model.load_weights(weight_path)
```
Check `src/fine_tuning.py` and `src/distilling.py` for *weights loading* examples.

## Default hyper-parameter settings

Unless otherwise specified, we use following default hyper-parameter settings.

Param|Value|Param|Value
:---|---:|:---|---:
Augmentation strength|0.1|Pre-training epochs|30
Augmentation strategy|AugSIM|Projection Head (100%)|4-1
Batch size|64|Projection Head (10%)|4-4
Early stopping patience|20|Projection Head (1%)|4-3
Embedding dimension|64|Model size|128 (4x)
Learning rate|5e-4|Temperature|0.1

## Cite

If you find our paper & code are useful for your research, please consider citing us 😘:

```bibtex
@inproceedings{xu2020ccgl, 
  author = {Xovee Xu and Fan Zhou and Kunpeng Zhang and Goce Trajcevski}, 
  title = {Graph Data Augmentation for Contrastive Cascade Learning}, 
  booktitle = {Submitted for review},
  year = {2020},
  pages = {1--14},
}
```

We also have a [survey paper](https://arxiv.org/abs/2005.11041) you might be interested:

```bibtex
@article{zhou2020survey,
  author = {Fan Zhou and Xovee Xu and Goce Trajcevski and Kunpeng Zhang}, 
  title = {A Survey of Information Cascade Analysis: Models, Predictions and Recent Advances}, 
  journal = {arXiv:2005.11041}, 
  year = {2020},
  pages = {1--41},
}
```

## Acknowledgment

We would like to thank [Xiuxiu Qi](https://qhemu.github.io/xiuxiuqi/), Ce Li, [Qing Yang](https://www.linkedin.com/in/庆-杨-43ba1a142), and Wenxiong Li for sharing their computing resources and help us to test the codes. We would also like to show our gratitude to the authors of [SimCLR](https://github.com/google-research/simclr) (and [Sayak Paul](https://github.com/sayakpaul)), [node2vec](https://github.com/eliorc/node2vec), [DeepHawkes](https://github.com/CaoQi92/DeepHawkes), and others, for sharing their codes and datasets. 

## Contact

For any questions (as well as request for the pdf version) please open an issue or drop an email to: `xovee at live.com`

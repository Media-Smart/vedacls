## Introduction
vedacls is an open source  classification toolbox based on PyTorch.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation
### Requirements

- Linux
- Python 3.7+
- PyTorch 1.1.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 9.0
- Python 3.7.3

### Install vedacls

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedacls python=3.7 -y
conda activate veda
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the vedacls repository.

```shell
git clone https://github.com/Media-Smart/vedacls.git
cd vedacls
vedacls_root=${PWD}
```

d. Install dependencies.

```shell
pip install -r requirements.txt
```

## Prepare data
The format of dataset supported by vedacls toolbox is as follows,

```python
Dataset
├── train
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── ...
├── val
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── ...
├── test
    ├── 0
    ├── 1
    ├── 2
    ├── ...
```

## Train

a. Config

Modify some configuration accordingly in the config file like `configs/resnet18.py`

b. Run

```shell
python tools/train.py configs/resnet.py
```

Snapshots and logs will be generated at `${vedacls_root}/workdir`.

## Test

a. Config

Modify some configuration accordingly in the config file like `configs/resnet18.py`

b. Run

```shell
python tools/test.py configs/resnet18.py path_to_resnet18_weights
```

## Contact

This repository is currently maintained by Chenhao Wang ([@C-H-Wong](http://github.com/C-H-Wong)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).

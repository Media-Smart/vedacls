## Introduction

vedacls is an open source classification toolbox based on PyTorch.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation
### Requirements

- Linux
- Python 3.6+
- PyTorch 1.4.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 10.2
- PyTorch 1.4.0
- Python 3.6.9

### Install vedacls

1. Create a conda virtual environment and activate it.

```shell
conda create -n vedacls python=3.6.9 -y
conda activate vedacls
```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

3. Clone the vedacls repository.

```shell
git clone https://github.com/Media-Smart/vedacls.git
cd vedacls
vedacls_root=${PWD}
```

4. Install dependencies.

```shell
pip install -r requirements.txt
```

## Prepare data
The catalogue structure of dataset supported by vedacls toolbox is as follows:

```shell
data/
├── train
│   ├── 0
│   │   ├── XXX.jpg
│   │     
│   ├── 1
│   ├── 2
│   ├── ...
│
├── val
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── ...
│ 
├── test
    ├── 0
    ├── 1
    ├── 2
    ├── ...
```

## Train

1. Config

Modify some configuration accordingly in the config file like `configs/resnet18.py`

2. Run

```shell
python tools/train.py configs/resnet18.py
```

Snapshots and logs will be generated at `${vedacls_root}/workdir/resnet18`

## Test

1. Config

Modify some configuration accordingly in the config file like `configs/resnet18.py`

2. Run

```shell
python tools/test.py configs/resnet18.py checkpoint_path
```

## Inference

1. Config

Modify some configuration accordingly in the config file like `configs/resnet18.py`

2. Run

```shell
python tools/inference.py configs/resnet18.py checkpoint_path image_path
```

## Deploy
1. Install volksdep following the [official instructions](https://github.com/Media-Smart/volksdep)

2. Benchmark(optional)
```shell
python tools/deploy/benchmark.py configs/resnet18.py checkpoint_path image_path
```
More available arguments are detailed in [tools/deploy/benchmark.py](https://github.com/Media-Smart/vedacls/blob/master/vedacls/tools/deploy/benchmark.py)

3. Export model
```shell
python tools/deploy/export.py configs/resnet18.py checkpoint_path image_path out_model_path
```
More available arguments are detailed in [tools/deploy/export.py](https://github.com/Media-Smart/vedacls/blob/master/vedacls/tools/deploy/export.py)

## Contact

This repository is currently maintained by Chenhao Wang ([@C-H-Wong](http://github.com/C-H-Wong)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).

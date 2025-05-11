## Introduction

The source code of the paper **GRUformer:A Novel Transformer Based Model for Automatic Modulation Recognition**

## Requirements

- pytorch>=2.0.1
- numpy
- matplotlib
- h5py (could run `pip install h5py` to install it)
- scikit-learn (could run `pip install scikit-learn` to install it)

## How to start

1. clone the repository
2. Install [Requirements](#requirements)
3. Download and unzip [RML2018.01A dataset](https://www.kaggle.com/datasets/pinxau1000/radioml2018/data)
4. Change the [path_data](./model_dataset.py#17) to dataset path
5. Run `python main.py` to train, and modify the [model_path](./main.py#22) when testing

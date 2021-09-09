# tobigs-image-conference
Video object segmentation branch  

base line code: [Siammask](https://github.com/foolwood/SiamMask)

## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Testing Models](#testing)
4. [Training Models](#training)

## Environment setup
This code has been tested on Ubuntu 18.04.5, Python 3.7, Pytorch 1.8.1, CUDA 10.2

- Clone the repository

base directory: `Siammask`
- Setup python environment
```shell
conda create -n siammask python=3.7
conda activate siammask
pip install -r requirements.txt
```

## Demo
- [Setup](#environment-setup) your environment
- Download the SiamMask model
```shell
cd $SiamMask
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Run `demo.py`

```shell
python demo.py --resume SiamMask_DAVIS.pth --config config_davis.json
```

<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask_demo.gif" width="500px" />
</div>

## Testing
- [Setup](#environment-setup) your environment
- Download test data
```shell
cd $SiamMask/data
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
ln -s ./DAVIS ./DAVIS2016
ln -s ./DAVIS ./DAVIS2017
```
- Download pretrained models
```shell
cd $SiamMask
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Evaluate performance on [DAVIS](https://davischallenge.org/) (less than 50s)
<br>
make `logs`directory for logging
```shell
CUDA_VISIBLE_DEVICES=0 python -u test.py --config config_davis.json --resume SiamMask_DAVIS.pth --mask --refine --dataset DAVIS2017 2>&1 | tee logs/test_$dataset.log
```

<br>
Note : You must run the code on GPU environment


## Training

### Training Data
- Go `data` directory and setup datas followed the `readme.md`, respectively
- Datasets we used are Youtube-VOS & COCO

### Download the pre-trained model (174 MB)
(This model was trained on the ImageNet-1k Dataset)
```shell
cd $SiamMask/experiments
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
ls | grep siam | xargs -I {} cp resnet.model {}
```

### Training SiamMask model with the Refine module
- [Setup](#environment-setup) your environment


- 경우1. You have base model's traubed parameter or checkpoint parameter
```shell
cd $SiamMask
bash python -u train.py --config=config.json -b 64 -j 20 --pretrained checkpoint_e12.pth --epochs 20 2>&1 | tee logs/train.log
```
- 경우2. Using parameter trained by paper's original author: `SiamMask_DAVIS.pth` for run demo
```shell
cd $SiamMask
bash python -u train.py --config=config.json -b 64 -j 20 --pretrained SiamMask_DAVIS.pth --epochs 20 2>&1 | tee logs/train.log
```
- You can view progress on Tensorboard (logs are at <experiment\_dir>/logs/)

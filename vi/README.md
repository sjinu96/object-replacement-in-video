# tobigs-image-conference
Deep Video Inpainting branch  

base line code: [Deep_Video_Inpainting](https://github.com/mcahny/Deep-Video-Inpainting)

## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)

## Environment setup
This code has been tested on Ubuntu 18.04.5, Python 3.7, Pytorch 1.8.1, CUDA 10.2, c++14  
If you don't use gnu c++14 complier, then you will encounter CUDA build error  

- Clone the repository  

- Setup python environment
```shell
git clone https://github.com/shkim960520/tobigs-image-conference.git
cd tobigs-image-conference
git checkout video_inpainting
conda create -n vinet python=3.7 -y
conda activate vinet
conda install cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt
bash install.sh
```

## Demo
- [Setup](#environment-setup) your environment
- Download the Deep Video Inpainting model
```shell
cd results/vinet_agg_rec

file_id="1_v0MBUjWFLD28oMfsG6YwG_UFKmgZ8_7"
file_name="save_agg_rec_512.pth"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

file_id="12TRCSwixAo9AqyZ0zXufNKhxuHJWG-RV"
file_name="save_agg_rec.pth"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

cd ../../
```
- Run `demo_vi.py`

```shell
python3 demo_vi.py 
```
- You can see the result under results folder

## TODO
1. Customizing Datalodaer for our trainding data (e.g. video, image etc)  
2. Fit the format of siammask output and our input  
3. Construct loss function  
4. Full stream of train model (e.g. train, save model, load model etc)  

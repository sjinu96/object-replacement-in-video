
### Final project : [erAIser](https://github.com/shkim960520/erAIser) - Remove an object in video using AI

[![file](https://img.shields.io/badge/presentation-file-orange)](https://drive.google.com/file/d/176qC2l3OYg_Uodj144VQRZ2MIT7Kobhu/view?usp=sharing)
[![video](https://img.shields.io/badge/presentation-video-orange)](https://www.youtube.com/watch?v=ebZxKIbMvqo)


# Moving Object Replacement in Video(called 'AANet' in full project)

본 프로젝트는 [github.com/snapresearch](https://github.com/snap-research/articulated-animation)의 모델을 기반으로 합니다.
해당 모델을 새로운 task(물체가 지워진 동영상 내에 새로운 애니메이션을 생성하는)을 적용하기 위해 추가한 모듈과 함수는 아래와 같습니다. 

[]()  
[]() 


## Code Files for Object Replacement
[aa_inference.py](https://github.com/sjinu96/erAIser/blob/main/AANet/aa_inference.py) : 전체적인 inference 과정을 포함하는 class AAInference  
[preprocessing_aanet.py](https://github.com/sjinu96/erAIser/blob/main/AANet/preprocessing_aanet.py) : inference에 필요한 함수들  
[preprocessing_davis.py](https://github.com/sjinu96/erAIser/blob/main/AANet/preprocessing_davis.py) : dataset의 정제를 위해 필요한 함수들.  
[]()  
[]() 


## Example



<table width="500" rules="cols" borderstyle="dashed" style="border:2px solid skyblue">
        <tr>
          <td>Original <img alt="" src="https://user-images.githubusercontent.com/71121461/132659892-60392b50-fe45-43b7-bca6-70561d3b04d9.gif" width=200 /></td><td>AANet <img alt="" src="https://images.velog.io/images/sjinu/post/8d2aee3e-1787-4f88-8c15-12be3ec75591/aanet_result_with_clipping_0.4_very_low_right_right.gif" width=200 /></td><td>Source  <img alt="" src="https://user-images.githubusercontent.com/71121461/132661327-19377651-e262-4b2b-abd8-d9cfc8b53f8c.png" width=100 ></td>
        <tr>
</table>

<table width="500" rules="cols" borderstyle="dashed" style="border:2px solid skyblue">
        <tr>
          <td>Original <img alt="" src="https://user-images.githubusercontent.com/71121461/132659821-34657c0c-6a68-4dc0-a05f-284cec4c81db.gif" width=250/></td><td>AANet <img alt="" src="https://images.velog.io/images/sjinu/post/8c6d11e3-334f-4ea5-beb0-851d6e91e647/Mike!.gif" width=150 /></td><td>Source <img alt="" src="https://user-images.githubusercontent.com/71121461/132661865-d455737a-6f77-4425-93da-9b5db4715c4f.png" width=100></td>
       <tr>
</table>


## Dataset Used for Training


![image](https://user-images.githubusercontent.com/71121461/132973929-6ccd1112-d7c0-4967-9b39-251ea637320b.png)


1. [DAVIS](https://davischallenge.org/davis2017/code.html)

2. [Taichi](https://paperswithcode.com/dataset/tai-chi-hd)

3. [TikTok](https://paperswithcode.com/dataset/tiktok-dataset)

4. [Cough](https://web.bii.a-star.edu.sg/~chengli/FluRecognition.htm)




# [Full Project(erAIser)](https://github.com/shkim960520/erAIser) and Implementation code

1. [erAIser](#erAIser)
2. [Example](#Example)
3. [Demo screenshot](Demo-screenshot)
4. [Usage](#Usage)
    - Environment setup
    - Run demo
5. [References](#References)
6. [Contributors](#Contributors)

## erAIser 
<br>
‘erAIser’ is a service that provides a specific object erased video by using video object segmentation and video inpainting methods.
<br>
<br>
Most of video inpainting model need segmentation mask of objects. But it is hard to get in normal way. For your convenience, we used a deep learning model that allows users to easily obtain segmentation masks. We combined this video object segmentation model with the video inpainting model to increase usability. Additionally, we use AANet to change object to other object in object erased video.
<br>
<br>
Our team consists of nine members of ‘Tobigs’ who are interested in computer vision task.
<br>
<br>
All you have to do is draw the object bounding box that you want to erase in the first frame. Then 'erAIser' will make a video of the object being erased. 
Let’s make your own video of a specific object being erased with ‘erAIser’!

<br>

## Example of erAIser(not using object replacement)
<br>

<table>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/68596881/126003086-60641341-2845-4a26-94ac-de0132a46cd4.gif" width = 400 height = 300/></td><td><img src="https://user-images.githubusercontent.com/68596881/126003512-ca14ddd5-a54c-4299-801e-3de3378fd8d3.gif" width = 400 height = 300/></td>
  <tr>
</table>

<br>

## Demo screenshot
<br>
<p align="center"><img width="980" alt="front" src="https://user-images.githubusercontent.com/68596881/126006048-1625a188-5e9c-478d-9787-ff8d09c3039a.png">
</p>

<br>
You can look around our web at the following link. For your information, the model is currently not working on the web due to the GPU environment.

[web link](https://eraiser-tobigs.netlify.app/).

## Usage

### Environment Setup
This code was tested in the following environments
 - Ubuntu 18.04.5
 - Python 3.7
 - Pytorch 1.8.1
 - CUDA 9.0
 - GCC 5.5 (gnu c++14 compiler)

If you don't use gnu c++14 compiler, then you will encounter CUDA build error  

1. Clone the repository & Setup

```bash
git clone https://github.com/sjinu96/object-replacement-in-video.git
cd object-replacement-in-video
conda create -n erAIser python=3.7 -y
conda activate erAIser
conda install cudatoolkit=9.0 -c pytorch -y
pip install -r requirements.txt
bash install.sh
```

2. Setup python path

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
cd vos/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../vi/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../web/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../AANet/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../
```

### Demo

1. Setup your [environment](#Environment-setup)
2. Download the Deep Video Inpainting model

```bash
cd vi/results/vinet_agg_rec

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

cd ../../../
```
3. Download the Siammask model

```bash
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth

file_id="1IKZWpMeWXq-9osBqG7e7bTABumIZ32gB"
file_name="checkpoint_e19.pth"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}
```
4. Download the AANet model
```bash
cd AANet/
mkdir checkpoints
cd checkpoints/

file_id="1DT6_SZHTkmuEWvCfs07F2mGLSkgxYplo"
file_name="4dataset384.pth"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

cd ../../
```

5. Make `results` directory for saving result video
```bash
mkdir results
```
`results` is defualt setting. You can change this.

6-1. Run `inference.py` for erasing
```bash
python3 inference.py --resume checkpoint_e19.pth --config config_inference.json
```

**6-2. Run `inference.py` for change object to other (ex. person, animation character)**
```bash
python3 inference.py --resume SiamMask_DAVIS.pth --config config_inference.json --using_aanet True
```
The result video will be saved in `results`.

</table>

---

**Following part is *copy and paste* from [original repository by snap-research](https://github.com/snap-research/articulated-animation)**

# Base Research for Object Replacement in video 

## Training for base model in AANet

To train a model run:
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --device_ids 0
```
The code will create a folder in the log directory (each run will create a time-stamped new folder). Checkpoints will be saved to this folder.
To check the loss values during training see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.
Then to train **Animation via disentaglement (AVD)** use:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --checkpoint log/{folder}/cpk.pth --config config/dataset_name.yaml --device_ids 0 --mode train_avd
```
Where ```{folder}``` is the name of the folder created in the previous step. (Note: use backslash '\' before space.)
This will use the same folder where checkpoint was previously stored.
It will create a new checkpoint containing all the previous models and the trained avd_network.
You can monitor performance in log file and visualizations in train-vis folder.


### Training on your own dataset
1) Resize all the videos to the same size, e.g 256x256, the videos can be in '.gif', '.mp4' or folder with images.
We recommend the latter, for each video make a separate folder with all the frames in '.png' format. This format is loss-less, and it has better i/o performance.

2) Create a folder ```data/dataset_name``` with 2 subfolders ```train``` and ```test```, put training videos in the ```train``` and testing in the ```test```.

3) Create a config file ```config/dataset_name.yaml```. See description of the parameters in the ```config/vox256.yaml```.  Specify the dataset root in dataset_params specify by setting  ```root_dir:  data/dataset_name```.  Adjust other parameters as desired, such as the number of epochs for example. Specify ```id_sampling: False``` if you do not want to use id_sampling.


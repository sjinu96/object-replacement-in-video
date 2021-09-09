# erAIser - Remove an object in video using AI
[![file](https://img.shields.io/badge/presentation-file-orange)](https://drive.google.com/file/d/176qC2l3OYg_Uodj144VQRZ2MIT7Kobhu/view?usp=sharing)
[![video](https://img.shields.io/badge/presentation-video-orange)](https://www.youtube.com/watch?v=ebZxKIbMvqo)
<p align="center"><img width="980" alt="첫슬라이드" src="https://user-images.githubusercontent.com/40483474/125912276-4d5b8952-7973-4884-80ff-93f475fb3bb8.PNG">
</p>

## Contents
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

## Example
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
git clone https://github.com/shkim960520/tobigs-image-conference.git
cd tobigs-image-conference
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

6-2. Run `inference.py` for change object to other (ex. person, animation character)
```bash
python3 inference.py --resume SiamMask_DAVIS.pth --config config_inference.json --using_aanet True
```
The result video will be saved in `results`.

## References
- Wang, Qiang, et al. "Fast online object tracking and segmentation: A unifying approach." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
- Wang, Tianyu, et al. "Instance shadow detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
- Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Deep video inpainting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5792–5801, 2019.
- Siarohin, Aliaksandr and Woodford, et al. "Motion Representations for Articulated Animation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

## Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/han811"><img src="https://user-images.githubusercontent.com/68596881/126450991-076c4c5c-a960-4372-a08c-0460fa9cbf11.png" width="150" height="150"><br /><sub><b>김태한 <br> Inpainting</b></sub></td>
    <td align="center"><a href="https://github.com/YoojLee"><img src="https://user-images.githubusercontent.com/68596881/126451113-e60a8c58-3fa0-4d7f-a42b-df3c3123a0d5.png" width="150" height="150"><br /><sub><b>이유진 <br> Inpainting, Web</b></sub></td>
    <td align="center"><a href="https://github.com/shkim960520"><img src="https://user-images.githubusercontent.com/68596881/126451157-a8aba5b6-71b4-400f-8b80-01b8a0d1345a.png" width="150" height="150"><br /><sub><b>김상현 <br> Video Object Segmentation, Web</b></sub></td>
    <td align="center"><a href="https://github.com/mink7878"><img src="https://user-images.githubusercontent.com/68596881/126451217-b2ef99bc-edfd-479c-bf06-81a745ea37e7.png" width="150" height="150"><br /><sub><b>김민경 <br> Video Object Segmentation</b></sub></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><a href="https://github.com/araseo"><img src="https://user-images.githubusercontent.com/68596881/126451283-7b86921c-219e-4f08-99bd-56a31cef96d9.png" width="150" height="150"><br /><sub><b>서아라 <br> Inpainting</b></sub></td>
    <td align="center"><a href="https://github.com/joy5075"><img src="https://user-images.githubusercontent.com/68596881/126451367-a16223af-f729-47e0-bdd7-f4b2bc67b743.png" width="150" height="150"><br /><sub><b>장혜림 <br> Video Object Segmentation</b></sub></td>
    <td align="center"><a href="https://github.com/5hyeonkwon"><img src="https://user-images.githubusercontent.com/68596881/126451450-e7f2f302-73ab-47fe-be6e-77fe29ad7a1e.png" width="150" height="150"><br /><sub><b>권오현 <br> Video Object Segmentation, Web</b></sub></td>
    <td align="center"><a href="https://github.com/sjinu96"><img src="https://user-images.githubusercontent.com/68596881/126451520-b89d3ce5-0286-443a-846c-547b70e0bd28.png" width="150" height="150"><br /><sub><b>박진수 <br> Inpainting, AAnet</b></sub></td>
    <td align="center"><a href="https://github.com/Jy0923"><img src="https://user-images.githubusercontent.com/68596881/126451575-73e3f251-05cc-44bc-a34f-9da0dbb41b2d.png" width="150" height="150"><br /><sub><b>오주영 <br> Inpainting, Web</b></sub></td>
  </tr>
</table>

import os

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import matplotlib.animation as animation


import imageio
import numpy as np

import PIL
### 전처리함수
def get_bbox(mask, mod='park'): # mask : Image type, mod='Park' : 내 bounding box 방법
    # get contours (presumably just one around the nonzero pixels) 
    if (mask!=0).sum()==0: # 바운딩 박스가 없을 시
        raise NameError('There is no bbox')
    
    if mod=='park': # array type
        
        ymin=np.where(mask!=0)[0].min()
        ymax=np.where(mask!=0)[0].max()
        xmin=np.where(mask!=0)[1].min()
        xmax=np.where(mask!=0)[1].max()
        
        return xmin, ymin, xmax-xmin, ymax-ymin
    
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
    return x,y,w,h

def test_bbox(images_path, mod='park'):
    imgs=[]
    if mod=='park':
        print('get bbox (mod : park)')
    for image_path in images_path:
        # read image as grayscale
        img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        x,y,w,h=get_bbox(img, mod)

        img=Image.fromarray(img)
        draw=ImageDraw.Draw(img)
        draw.rectangle(((x, y),(x+w, y+h)), width=13, outline='#00ff00cc')
        imgs.append(img)
    showImagesHorizontally(imgs, show_by_image=True)
    
def draw_bbox(img, bbox):
    # bbox=(x,y,w,h)
    x,y,w,h=bbox
    img=PIL.Image.fromarray((img*255).astype(np.uint8))
    draw=PIL.ImageDraw.Draw(img)
    print(x,y,w,h)
    draw.rectangle([(x,y) ,(x+w, y+h)], width=10, outline=(0, 0, 0))

    plt.imshow(np.array(img))
    
def showImagesHorizontally(list_of_files, show_by_image=False, width=8, title_of_files=None, start=None):
    fig = figure(figsize=(width*2,width*2))
    number_of_files = len(list_of_files) # list_of_files=(image_list)
    for i in range(number_of_files)[:width*width]:
        a=fig.add_subplot(width,width,i+1)
        if show_by_image:
            if title_of_files:
                plt.title(title_of_files[i] +f'-{start+i}')
            imshow(list_of_files[i])
        else:
            if title_of_files:
                plt.title(title_of_files[i] +f'-{start+i}')
            image = imread(list_of_files[i])
            imshow(image,cmap='Greys_r')
        axis('off')
    plt.tight_layout()

    
    
def bbox2squared(bbox, sqaure_size, image_size):

    x,y,w,h=bbox
    xmin, xmax, ymin, ymax= (x, x+w, y, y+h)  # 이 때 w,h는 bounding box의 크기이다.
    width,height=image_size
    # center 
    cx = (xmin+xmax)//2
    cy = (ymin+ymax)//2
    r = sqaure_size//2 # for int.

    #  늘려서 잘릴 경우.
    if cx-r<0: # 왼쪽 잘림의 경우
        cx=r
    if cx+r>width: # 오른쪽 잘림의 경우
        cx=width-r # 단, r<cx<w-r 이여야 한다. 즉, 2r<w
    if cy-r<0: # 위쪽 잘림의 경우
        cy=r
    if cy+r>height: # 아래쪽 잘림의 경우
        cy=height-r # 마찬가지로 2r<h여야 하는데 그렇지 못할 것.

    if (cx-r<0) or (cx+r)>width or (cy-r)<0 or (cy+r)>height:
        print(width, height, cx, cy, r)
        raise NameError('Bounding box 처리 오류 (이미지를 벗어남)')

    return np.array([cx-r, cx+r, cy-r, cy+r])


def crop_image_by_bbox(image, annotation, bbox, width, height,
                       mod='park', return_mask=True, train_size=384, crop_size=384,):
    
#     train_size=train_size
# #     crop_size=crop_size
#     width,height=854, 480 # 이 때 w,h는 전체 image의 크기이다.
    
    new_xmin, new_xmax, new_ymin, new_ymax=bbox2squared(bbox, crop_size, (width, height))
    new_image=image[new_ymin:new_ymax, new_xmin:new_xmax, :]
    new_annotation=annotation[new_ymin:new_ymax, new_xmin:new_xmax]
    
    if train_size < crop_size:
        new_image=cv2.resize(new_image, (train_size, train_size))
        new_annotation=cv2.resize(new_annotation, (train_size, train_size))
    if return_mask==True: # 
        return new_image, new_annotation
    else:
        return new_image

    
def crop_video_using_path(video_dir, anno_dir, masking=False, 
                          return_original=False, mod='park', expand_bbox=False): # annotation이 존재하는 데이터의 경우에만 사용 가능하다. 
    
    
    train_size=384 # 384 x 384. train_size<480.
    crop_size=train_size
    
    width,height=854, 480
    images_path=glob.glob(os.path.join(video_dir, '*.jpg'))
    annos_path=glob.glob(os.path.join(anno_dir, '*.png'))
    # 이미지 순서 정렬
    images_path.sort()
    annos_path.sort() 
    
    new_images=[]
    
    #일단 다 받아보자.
    if return_original==True:
        images=[]
        annotations=[]
        new_annotations=[]
        bboxes=[]
    
    if expand_bbox:
        for i in range(len(images_path)):
            anno_image=cv2.imread(annos_path[i], cv2.COLOR_RGB2GRAY)
            
            try:
                x,y,w,h=get_bbox(anno_image, mod=mod)
            except NameError as e: # e : 'no bbox in image', bounding box가 없는 경우 get_bbox에서 오류가 뜰 수 있다.
                print(e, annos_path[idx])
                pass # 이전 frame의 좌표 사용
            
            if w > train_size or h > train_size:
                crop_size=min((width, height)) # 거의 480
#                 print('expand bbox to size ', crop_size)
                break
            
    for idx, image_path in enumerate(images_path):
        
        image=plt.imread(image_path)
        anno_image=cv2.imread(annos_path[idx], cv2.COLOR_BGR2GRAY)
        
        try:
            x,y,w,h=get_bbox(anno_image, mod) # 바운딩 박스의 xmin, ymin과 크기.
        except NameError as e: # e : 'no bbox in image', bounding box가 없는 경우 get_bbox에서 오류가 뜰 수 있다.
            print(e, annos_path[idx])
            pass # 이전 frame의 좌표 사용
        
        if masking==True:
            new_image, new_annotation=crop_image_by_bbox(image, anno_image, bbox=(x,y,w,h), train_size=train_size, crop_size=crop_size,  return_mask=masking)

            new_image=mask_image(new_image, new_annotation)
            new_images.append(new_image)
        else:
            new_image=crop_image_by_bbox(image, anno_image, bbox=(x,y,w,h), train_size=train_size,crop_size=crop_size, return_mask=masking)
            new_images.append(new_image)
            
        if return_original==True:
            images.append(image)
            annotations.append(anno_image)
            new_annotations.append(new_annotation)
            bboxes.append(bbox2squared([x,y,w,h], train_size, (width, height)))
            
    if return_original==True:
        return new_images, images, annotations, new_annotations, bboxes
    else: 
        return new_images   # 반환은 np.array형태로.
     

def mask_image(image, annotation, reverse=False):

    # 255 to 1
    if np.any(image>1):
        image=image/255
        
    
    if len(annotation.shape)==3:
        annotation=128*((annotation[:, :, 0]!=0) | (annotation[:, :, 1]!=0)|(annotation[:, :, 2]!=0)).astype(int)
        if len(annotation.shape)==3:
            raise NameError('dimageension of annotation is 3')
    annotation=annotation.astype('uint8')
    if reverse==False:
        object_mask=(annotation!=0).astype(int) # mask to 1
        background_mask=1*(annotation==0).astype(int) # Background to 255
    if reverse==True:
        object_mask=(annotation==0).astype(int) # mask to 1
        background_mask=1*(annotation!=0).astype(int) # Background to 255        
    # 1d to 3d
    object_mask=np.broadcast_to(object_mask[..., np.newaxis], annotation.shape+(3,))#.astype('uint8')
    background_mask=np.broadcast_to(background_mask[..., np.newaxis], annotation.shape+(3,))#.astype('uint8')
    
#     print(image.shape, object_mask.shape, background_mask.shape)
    new_image=(image*object_mask+background_mask)
    new_image=np.array(new_image)
    return new_image
    
def display_aaresult(source, driving, generated=None, save=False):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])
    if save==True:
        return ims
    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani
    
def display_video(images_array):
    fig = plt.figure(figsize=(6,6))

    ims = []
    for i in range(len(images_array)):
        im = plt.imshow(images_array[i], animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani

def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


def synthesize_prototype(original_video, predictions, annotations, new_annotations, bboxes):
    new_video=[]
    i=0

    for frame in range(len(original_video)):
        frame_only_background=mask_image(original_video[frame], annotations[frame], reverse=True)
        #frame_onlybboxes[frame]

        ymin, ymax, xmin, xmax=bboxes[frame][2],bboxes[frame][3],bboxes[frame][0],bboxes[frame][1]

        part=frame_only_background[ymin:ymax, xmin:xmax]

        pixel_mean=new_annotations[frame].mean(axis=2)
        mask_remover=np.broadcast_to(pixel_mean[..., np.newaxis], pixel_mean.shape+(3,)).astype('uint8')

        part=part*(mask_remover==0).astype(int)

        a=mask_image(predictions[frame], new_annotations[frame])
        a[a==255]=0

        a=(a*255).astype(int)

        part=part+a

        frame_only_background[ymin:ymax, xmin:xmax]=part

        new_video.append(frame_only_background)
    return new_video

def resize_bbox_using_pad_white(bbox, im,  pad_const=0):
    # 단, 이 경우는 object bbox에 직접적으로 적용되는 코드다.(즉, (x,y)=(0,0)이라 가정한다.)
    # 이는, input image의 size invariance를 처리하기 위해, object에만 집중하기 위함.
    # 정사각형, 정사각형 패딩, 비율 변경(0710 해야함)까지.
    
    x,y, width, height=bbox
    
    im_array=np.array(im)
    h,w=im_array.shape[:2]
#     print(h,w)
    pad_length=abs(w-h)  
#     print(pad_length)

    if pad_const:
        add_pad_length=pad_const
#         add_pad_length=pad_const-np.max([w,h])
        total_pad_length=add_pad_length+pad_length
    else:
        total_pad_length=pad_length
    
    if w>h:
        if not pad_const:
            return  [0, total_pad_length//2, width, height]
        if pad_const:
            return [add_pad_length//2, total_pad_length//2, width, height] 
    else:
        if not pad_const:
            return [total_pad_length//2, 0, width, height]
            
        if pad_const:
            return [total_pad_length//2, add_pad_length//2, width, height]
        
        
        
        
    
    
    
        
def resize_bbox(bbox, prior_aspect, post_aspect):
    prior_w, prior_h=prior_aspect
    post_w, post_h=post_aspect
    
    x,y,w,h=bbox
    
    
    x_scale=post_w/prior_w
    y_scale=post_h/prior_h
    
    
    new_x = int(x_scale*x)
    new_y = int(y_scale*y)
    new_w=int(x_scale*w)
    new_h=int(y_scale*h)
    
    return [new_x, new_y, new_w, new_h]
# 정사각형, 정사각형 패딩, 비율 변경(0710 해야함)까지.
def pad_white(im, pad_const=False): # im : imageio.imread로 읽은...(0~1)
    im_array=np.array(im)
    h,w=im_array.shape[:2]
    pad_length=abs(w-h)  
    
    if pad_const:
        add_pad_length=pad_const
        total_pad_length=add_pad_length+pad_length
    else:
        total_pad_length=pad_length
    
    pad_const2=np.max([w,h])+pad_const
#     print(pad_const2)
#     print(add_pad_length)
    
    if w>h:
        upper=np.full(((total_pad_length)//2, w, 3),1)
        lower=np.full((total_pad_length-total_pad_length//2, w, 3), 1)
        
        im_first=np.concatenate([upper, im_array, lower], axis=0)
        if not pad_const:
            return imageio.core.util.Image(im_first)
        if pad_const:
            left=np.full((pad_const2,add_pad_length//2, 3),1)
            right=np.full((pad_const2,add_pad_length-add_pad_length//2, 3),1)
        
            return imageio.core.util.Image(np.concatenate([left, im_first, right], axis=1))
    else:
        left=np.full((h,total_pad_length//2, 3),1)
        right=np.full((h,total_pad_length-total_pad_length//2, 3),1)
        
        im_first=np.concatenate([left, im_array, right], axis=1)

        if not pad_const:
            return imageio.core.util.Image(im_first)
        if pad_const:
            upper=np.full((add_pad_length//2,pad_const2, 3),1)
            lower=np.full((add_pad_length-add_pad_length//2,pad_const2, 3),1)
            return imageio.core.util.Image(np.concatenate([upper, im_first, lower], axis=0))
  



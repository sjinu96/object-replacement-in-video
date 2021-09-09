import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import matplotlib.animation as animation
from IPython.display import HTML
import imageio
import numpy as np
from skimage import img_as_ubyte
from preprocessing_aanet import *


def crop_video_using_path(video_dir, anno_dir, masking=False, 
                          return_original=False, mod='park', expand_bbox=False): # annotation이 존재하는 데이터의 경우에만 사용 가능하다. 
    
    
    train_size=384 # 384 x 384. train_size<480.
    crop_size=train_size
    
    width,height=854, 480
    images_path=glob(os.path.join(video_dir, '*.jpg'))
    annos_path=glob(os.path.join(anno_dir, '*.png'))
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
    
def convert_ds_to_squared_video(train_ds_dir, train_save_dir=False, test_save_dir=False, test=False, save=False, sp=0,  masking=False, inference=False, expand_bbox=True):
    print('Masking : ', str(masking), '...')
    train_ds_anno_dir=os.path.join(train_ds_dir, sub_anno_dir)
    train_ds_image_dir=os.path.join(train_ds_dir, sub_image_dir)
    
    assert(os.listdir(train_ds_anno_dir)==os.listdir(train_ds_image_dir))

    train_ds_videos=os.listdir(train_ds_anno_dir)

    train_num=len(train_ds_videos)-len(train_ds_videos)//20 # 5%만 test로..
    
    if test==False:
        print(f'start, for train, numer of video : {train_num}')
        videos=train_ds_videos[sp:train_num]
        save_dir=train_save_dir
    else:
        print(f'start, for test, video : {len(train_ds_videos)-train_num}')
        videos=train_ds_videos[sp+train_num:]
        save_dir=test_save_dir
    count=0
    
        
    for i, video in enumerate(videos):
        print(sp+i, '번째 video -->', video)
        if save==False:
            if i!=inference:
                continue
        
        count+=1
        if count%5==0:
            if test==False:
                print(f'processing...({count}/{train_num})')
            else:
                print(f'processing...({count}/{len(train_ds_videos)//20})')
                
        train_ds_image_ex_dir=os.path.join(train_ds_image_dir, video)
        train_ds_anno_ex_dir=os.path.join(train_ds_anno_dir, video)
        #print(os.path.join(train_ds_image_dir, video))
        #print(os.path.join(train_ds_anno_dir, video))
        
        train_ds_images_path=glob(os.path.join(train_ds_image_ex_dir, '*.jpg'))
        train_ds_images_path.sort()
#         train_ds_annos_path=glob(os.path.join(train_ds_anno_ex_dir, '*.png'))
        train_ds_images_name=[path.split('/')[-1].split('.')[0] for path in train_ds_images_path]
#        print(train_ds_anno_ex_dir)
        new_images=crop_video_using_path(train_ds_image_ex_dir, train_ds_anno_ex_dir, mod='park', masking=masking, expand_bbox=expand_bbox)
        
        if save==False:
            return new_images
        
        # 저장
        if not os.path.exists(os.path.join(save_dir, video)):
            os.mkdir(os.path.join(save_dir, video))

        # train
        for idx, new_image in enumerate(new_images):
            imageio.imwrite(os.path.join(save_dir, video, train_ds_images_name[idx]+'.png'), new_image)
#             plt.imsave(os.path.join(save_dir, video, train_ds_images_name[idx]+'.png'), new_image)
        if count%5==0:
            print('save... --> ', os.path.join(save_dir, video))

            
            
# New instance dataset of DAVIS
def convert_ds_to_squared_video_by_instance(train_ds_dir, train_save_dir=False, test_save_dir=False, 
                                            test=False, save=False, sp=0,  masking=False, inference=False, expand_bbox=True):
    
    
    print('Masking : ', str(masking), '...')
    train_ds_anno_dir=os.path.join(train_ds_dir, sub_anno_dir)
    train_ds_image_dir=os.path.join(train_ds_dir, sub_image_dir)
    
    # Human folder와 맞춰주기.
    new_train_ds_image_dir=[]
    for image_dir in os.listdir(train_ds_image_dir):
        if image_dir in os.listdir(train_ds_anno_dir):
            new_train_ds_image_dir.append(image_dir)
    
#     new_train_ds_image_dir.sort()
    
    assert(new_train_ds_image_dir==os.listdir(train_ds_anno_dir))

    train_ds_videos=os.listdir(train_ds_anno_dir)
    
    
    
    train_num=len(train_ds_videos)-len(train_ds_videos)//20 # 5%만 test로..
    
    if test==False:
        print(f'start, for train, numer of video : {train_num}')
        videos=train_ds_videos[sp:train_num]
        save_dir=train_save_dir
    else:
        print(f'start, for test, video : {len(train_ds_videos)-train_num}')
        videos=train_ds_videos[sp+train_num:]
        save_dir=test_save_dir
    count=0
    
        
    for i, video in enumerate(videos):
        print(sp+i, '번째 video -->', video)
        if save==False:
            if i!=inference:
                continue
        
        count+=1
        if count%5==0:
            if test==False:
                print(f'processing...({count}/{train_num})')
            else:
                print(f'processing...({count}/{len(train_ds_videos)//20})')
                
        train_ds_image_ex_dir=os.path.join(train_ds_image_dir, video)
        train_ds_anno_ex_dir=os.path.join(train_ds_anno_dir, video)
        #print(os.path.join(train_ds_image_dir, video))
        #print(os.path.join(train_ds_anno_dir, video))
        
        train_ds_images_path=glob(os.path.join(train_ds_image_ex_dir, '*.jpg'))
        train_ds_images_path.sort()
#         train_ds_annos_path=glob(os.path.join(train_ds_anno_ex_dir, '*.png'))
        train_ds_images_name=[path.split('/')[-1].split('.')[0] for path in train_ds_images_path]
#        print(train_ds_anno_ex_dir)

        new_videos=crop_video_using_path_by_instance(train_ds_image_ex_dir, train_ds_anno_ex_dir, mod='park', masking=masking, expand_bbox=expand_bbox)
        
        if save==False:
            return new_videos

        instance_videos=os.listdir(train_ds_anno_ex_dir)

        for num_instance, instance_video in enumerate(instance_videos):

            # 저장
            if not os.path.exists(os.path.join(save_dir, instance_video)):
                 os.mkdir(os.path.join(save_dir, instance_video))

            for idx, new_image in enumerate(new_videos[num_instance]):
                imageio.imwrite(os.path.join(save_dir, instance_video, train_ds_images_name[idx]+'.png'), new_image)
            #             plt.imsave(os.path.join(save_dir, instance_video, train_ds_images_name[idx]+'.png'), new_image)
            if count%5==0:
                print('save... --> ', os.path.join(save_dir, instance_video))
            
            

def crop_video_using_path_by_instance(video_dir, annos_dir, masking=False, 
                          return_original=False, mod='park', expand_bbox=False): # annotation이 존재하는 데이터의 경우에만 사용 가능하다. 
    # anno_dir 내부에 sub directory들이 존재한다. 
    print(annos_dir)
    new_videos=[]
    for instance in range(len(os.listdir(annos_dir))):
        anno_dir=glob(os.path.join(annos_dir, '*'))[instance]
        train_size=384 # 384 x 384. train_size<480.
        crop_size=train_size

        width,height=854, 480
        images_path=glob(os.path.join(video_dir, '*.jpg'))
        annos_path=glob(os.path.join(anno_dir, '*.png'))
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
#                 print(annos_path)
                anno_image=cv2.imread(annos_path[i], cv2.COLOR_RGB2GRAY)

                try:
                    x,y,w,h=get_bbox(anno_image, mod=mod)
                except NameError as e: # e : 'no bbox in image', bounding box가 없는 경우 get_bbox에서 오류가 뜰 수 있다.
                    print(e, annos_path[i])
                    pass # 이전 frame의 좌표 사용

                if w > train_size or h > train_size:
                    crop_size=min((width, height)) # 거의 480
                    print('expand bbox to size ', crop_size)
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
        new_videos.append(new_images)
    if return_original==True:
        return new_videos, images, annotations, new_annotations, bboxes
    else: 
        return new_videos   # 반환은 np.array형태로.
    


def save_person(train_dir, mode='person1', plus4_save_eigen=False, save_dir=False, plus4_inference=False):
    # mode : person1, people2, people3
    # category name
    cat_name=train_dir.split('/')[-1]
    images_path=glob(os.path.join(train_dir, '*'))
    images_path.sort()
    
    ims=[]
    ims_names=[]
    for image_path in images_path:
        im_name=image_path.split('/')[-1]
        ims_names.append(im_name)
        
        im=cv2.imread(image_path)
        ims.append(im)
    
    new_ims=[]
    if mode=='person1':
        new_ims=ims
    elif mode=='people2':
        for im in ims:
            new_im=prep_2person(im)
            new_ims.append(new_im)
    elif mode=='people3':
        remove_channel, eigen_values=cal_objectnum(ims, mode='people3')
        for im in ims:
            new_im=prep_3person(im, remove_channel, eigen_values)
            new_ims.append(new_im)
                           
    elif mode=='people_many':
        new_eigens=[]
        len_eigens=[]
        eigen_values=cal_objectnum(ims, mode='people4')
        for im in ims:
            if plus4_inference:
                new_im, new_eigen, len_eigen=prep_4plus_people(im, eigen_values, plus4_save_eigen=plus4_save_eigen, inference=plus4_inference, 
                                                              save_dir=save_dir)
                new_ims.append(new_im)
                new_eigens.append(new_eigen)
                len_eigens.append(len_eigen)
                
            else:
                new_im=prep_4plus_people(im, eigen_values, plus4_save_eigen=plus4_save_eigen, save_dir=save_dir)
                new_ims.append(new_im)

    if save_dir==False:
        if plus4_inference:
            print('mode : ', mode, 'name : ', cat_name, ' ; no save - return : images ', 'plus4 inference : ', plus4_inference)
            print('len(ims): ' ,len(ims))
            return new_ims, new_eigens, len_eigens
        else:
            print('mode : ', mode, 'name : ', cat_name, ' and, no save - return : images ')
            return new_ims
        
    else:
        for i in range(len(new_ims[0])) : # i : i번째 사람
            # person 1
            save_sub_dir=os.path.join(save_dir, cat_name, cat_name+'_{}'.format(i+1))

            if not os.path.exists(save_sub_dir):
                os.mkdir(save_sub_dir)

            print('saving...', save_sub_dir)
            for idx, im in enumerate(new_ims):
                imageio.imwrite(os.path.join(save_sub_dir, ims_names[idx]) , im[i])

                
    return


def cal_objectnum(ims, mode='people4'):
    
    if mode=='people3':
        remove_channel=[1,2]
        set_eigen=set()
        for im in ims:
            combi=(im[:, :, remove_channel[0]]-1/2*im[:, :, remove_channel[1]])
            # 조합
            eigen_values=np.unique(combi)
            set_eigen=set_eigen.union(eigen_values)
        
        eigen_values=list(set_eigen)
        #print(eigen_values)
        eigen_values.remove(0) # 0은 제외한다.
        
        return remove_channel, eigen_values
    
    elif mode=='people4':

        set_eigen=set()
        for im in ims:
            combi=-im[:, :, 0]+1/3*im[:, :, 1]-1/6*im[:,:,2]
            # 조합
            eigen_values=np.round(np.unique(combi), 3)
            set_eigen=set_eigen.union(eigen_values)
        eigen_values=list(set_eigen)
        eigen_values.remove(0)

        return eigen_values 
    
    
def prep_1person(im):
    return im
def prep_2person(im):
    
    channel=[0,1,2]
    remove_channel=[]
    for chan in channel:
        if len(np.unique(im[:, :, chan]))==2: # mask가 존재하는 채널(0과 128)
            remove_channel.append(chan)
    #print(remove_channel) : # 1,2
    im_people=[]
    
    person1=im.copy()
    person1[:, :, remove_channel[0]]=0
    
    person2=im.copy()
    person2[:, :, remove_channel[1]]=0
    
    im_people.append(person1)
    im_people.append(person2)
    
    if len(im_people[0].shape)<3:
        raise NameError('Image should be 3d')
    return im_people
    
    
def prep_3person(im, remove_channel, eigen_values):    
    
    
    #print(remove_channel) # 1,2
    
    combi=(im[:, :, remove_channel[0]]-1/2*im[:, :, remove_channel[1]])
    # 조합
#     eigen_values=np.unique(combi)
#     eigen_values=np.delete(eigen_values, np.where(eigen_values == 0)) # 0은 제외한다.
    
    # 고유값이 3개가 아닌 경우
    if len(eigen_values)!=3:
        print(eigen_values)
        raise NameError('not 3person')
  
    im_people=[]
    
    for i in range(len(eigen_values)):
        people=combi.copy()
        people[people!=eigen_values[i]]=0
        people[people==eigen_values[i]]=128
        
        # 1d to 3d
        people=np.concatenate([np.zeros(test.shape)[..., np.newaxis], 
                               np.zeros(test.shape)[..., np.newaxis], people[..., np.newaxis]], axis=2).astype('uint8')
        im_people.append(people)

    if len(im_people[0].shape)<3:
        raise NameError('Image should be 3d')
        
    return im_people

def prep_4plus_people(im, eigen_values, plus4_save_eigen=False, inference=False, save_dir=False):

    
    combi=np.round(-im[:, :, 0]+1/3*im[:, :, 1]-1/6*im[:,:,2], 3)
#     print(np.unique(combi))
#     print(eigen_values)
    # 조합
#     eigen_values=np.unique(combi)
#     eigen_values=np.delete(eigen_values, np.where(eigen_values == 0)) # 0은 제외한다.

    im_people=[]
    eigen_people=[]
    for i in range(len(eigen_values)):
        people=combi.copy()
        people[people!=eigen_values[i]]=0
        people[people==eigen_values[i]]=128
          # 1d to 3d
        people=np.concatenate([np.zeros(people.shape)[..., np.newaxis], 
                               np.zeros(people.shape)[..., np.newaxis], people[..., np.newaxis]], axis=2).astype('uint8')
        #print(people.shape)
#         if len(np.unique(people))==1:
#             raise NameError('people mask\'s length is 1')
        
        if inference:
            if save_dir:
                raise NameError('can\'t use inference and save_dir simultaneously')
            
            eigen_people.append(eigen_values[i])
            im_people.append(people)
            
        else:
            if i in plus4_save_eigen:
                im_people.append(people)
    
    if len(im_people[0].shape)<3:
        raise NameError('Image should be 3d')
        
    if inference:
        return im_people, eigen_people, len(eigen_values)
    else:
        return im_people

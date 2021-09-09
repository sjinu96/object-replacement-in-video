import warnings
warnings.filterwarnings("ignore")
import os
from demo import make_animation
from demo import load_checkpoints
from skimage import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgba2rgb
import copy
import imageio

# 기타 필요 함수들.
from .preprocessing_aanet import *

class AAInference:
    
    def __init__(self, config_path, model_path): # 후에 args로 넣을 수도
        
        self.generator, self.region_predictor, self.avd_network = load_checkpoints(config_path=config_path,
                                                            checkpoint_path=model_path)
        
        self.source_image=None
        self.source_image_bbox=[]
        self.source_object_image=None
        self.source_image_384=None
        self.source_image_384_bbox=[]
        
        
        
        self.origin_video=[]
        self.origin_video_pos=[]
        self.origin_video_sz=[]
        self.origin_video_mask=[]
        self.origin_video_bbox=[]

        self.origin_video_512=[]
        self.origin_video_pos_512=[]
        self.origin_video_sz_512=[]
        self.origin_video_mask_512=[]
        self.origin_video_bbox_512=[]
        
        self.origin_video_expand_bbox=[]
        self.vi_result=[]
        
        self.driving_video=[]
        self.driving_video_384=[]
        self.driving_video_384_mask=[]
        
        # 인물 비율 테스트용
        self.keep_aspect_ratio=True
        self.aspect_ratio=1
        self.width_ratio=1
        self.for_test=[]
    
    
    def generate_driving_video_384(self, origin_video, origin_video_mask, origin_video_pos, origin_video_sz): # annotation이 존재하는 데이터의 경우에만 사용 가능하다. 

        train_size=384 # 384 x 384. train_size<480.
        crop_size=np.array(origin_video_sz).max()
        if train_size>crop_size:
            crop_size=train_size

        height,width=origin_video[0].shape[:2]

        new_driving_video=[]
        new_driving_video_mask=[]
        bboxes=[]

        for f in range(len(origin_video)):
            image=origin_video[f]
            anno_image=origin_video_mask[f]
            w,h=origin_video_sz[f]
            x,y=origin_video_pos[f]-(self.origin_video_sz[f]/2).astype(int)


            new_image, new_annotation=crop_image_by_bbox(image, anno_image, [x,y,w,h], width, height,
                                                         train_size=train_size, crop_size=crop_size)

            new_image=mask_image(new_image, new_annotation)
            new_driving_video.append(new_image)
            new_driving_video_mask.append(new_annotation)
            bboxes.append(bbox2squared([x,y,w,h], train_size, (width, height)))

        return new_driving_video, new_driving_video_mask, bboxes
    
    def resize_for_generating_animation(self, source_image, driving_video, source_image_bbox, pad_ratio=0.25):
        # AAInf.source_image가 직사각형인 경우, driving video와 우선 크기를 맞춰주자. 
        # keep_aspect_ratio : object의 비율 유지
        # else : 비율 변경(후술)
        
        ######
        # driving_video가 직사각형이라면 generating_driving_video 선행    
        ######
        
        x,y,w,h=source_image_bbox
        
        if source_image.shape[:2]!=driving_video[0].shape[:2]:
#             print('source image is not square')
            # 비율 유지하면서 resize.. 
            if self.keep_aspect_ratio:
#                 print('keep aspect ratio of original image')
                
                post_size=driving_video[0].shape[:2][::-1]
                pad_const=int(pad_ratio*max([w,h])) # 20%의 여유
                self.source_object_image=source_image[y:y+h, x:x+w]
                source_image_pad=pad_white(self.source_object_image, pad_const=pad_const)
                source_image=resize(source_image_pad, post_size)
                # bbox 변환
                
                self.source_image_384_bbox=resize_bbox_using_pad_white(source_image_bbox, self.source_object_image, pad_const)
                prior_size=source_image_pad.shape[:2] # pad_white의 결과 정사각형으로 나온다.
                self.source_image_384_bbox=resize_bbox(self.source_image_384_bbox, prior_size, post_size)
                
            else:
                # 0712 : 거의 안 쓸 것. 왜냐하면, 생성 후에 합성할 때 맞춰주면 그만이다. using ratio input over (512,512)
                print('change aspect ratio of object in source image')
                raise NameError('dont use change aspect ratio')
#                 # 0710 비율을 앞에 바운딩박스에서 가져올 수 있음. 비율 + 사이즈까지. 혹은 바운딩박스를 받을까
#                 new_size=tuple((np.array(driving_video[0].shape[:2])*self.aspect_ratio).astype(int))
#                 if self.width_ratio:
#                     new_size=(new_size[0], int(new_size[0]*self.width_ratio))
#                 source_image=pad_white(resize(source_image, new_size), pad_const=driving_video[0].shape[0])
# #                 source_image=resize(source_image, driving_video[0].shape[:2])
        
        # resize (아직 rgba 처리 안함.)
        # 어차피 위에서 driving video가 (384,384) 라면 실행 하나 마나이긴 하다.(RGBA 방지일뿐.)
        source_image = resize(source_image, (384, 384))[..., :3]
        self.source_image_384_bbox=resize_bbox(self.source_image_384_bbox, post_size, (384,384))
        
        driving_video = [resize(frame, (384, 384))[..., :3] for frame in driving_video]
        
        self.source_image_384=source_image
        self.driving_video_384=driving_video
        
        return source_image, driving_video
    def resize_animation(self, source_animation, origin_video, bbox_384):
        # source_animation : (384, 384)
        result_shape=(512,512)
        height,width= origin_video[0].shape[:2] 

    #     mean_w=np.array( oirgin_sz[:, 0])
    #     mean_h=np.array( oirgin_sz[:, 1])

        if width>=height:
            # 길이 가로인 동영상만..ㅠ..
            if width*0.85>height:
                width=int(width*0.85) # 덜 깎자. 

            new_width=int(384*(height/width))
            new_animation=[resize(pad_white(resize(f, (384, new_width))), (384,384)) 
                           for f in  source_animation]

            x,y,w,h=bbox_384
            cx, cy=int(x+w/2), int(y+h/2)
            new_w=int(w*(height/width))
            new_x=int(cx-new_w/2)
            new_bbox=[new_x, y, new_w, h]

            return [new_animation, new_bbox]


        else:
            new_height=int(384*(width/height))
            new_animation=[resize(pad_white(resize(f, (new_height, 384))), (384,384)) 
                           for f in  source_animation]

            x,y,w,h=bbox_384
            cx, cy=int(x+w/2), int(y+h/2)
            new_h=int(w*(width/height))
            new_y=int(cy-new_h/2)
            new_bbox=[x, new_y, w, new_h]

            return [new_animation, new_bbox]
        
    def resize_target_size(self, target_sizes, origin_video_sz_512, source_animation):
        ani_size=int(np.array(target_sizes)[:, 1].mean())
        ob_size=int(np.array(origin_video_sz_512)[:, 1].mean())

        if ani_size>ob_size:
            new_size=int(384*(ob_size/ani_size))
        else : 
            new_size=380
            ob_size=380
            ani_size=384

        new_target_sizes=[]
        for sz in target_sizes:
            new_sz=(sz*(ob_size/ ani_size)).astype(int)
            new_target_sizes.append(new_sz)
        
        pad_const=384-new_size
        new_animation=[]
        for f in source_animation:
            f_small=resize(f, (new_size, new_size))
            f_384=pad_white(f_small, pad_const=pad_const)
            new_animation.append(f_384)
            
        return new_target_sizes, new_animation
    
    def generate_animation(self, source_image, driving_video, ani_mode):

        
        predictions = make_animation(source_image, driving_video, self.generator, 
                              self.region_predictor, self.avd_network, animation_mode=ani_mode) # animation_mode 
        
        
        self.for_test=[source_image, driving_video, predictions]
        
        
        return predictions
    
    def decouple_background(self, source_animation, method='naive_filtering', filter_ratio=0.88):
        
        masks=[] 
        if method=='naive_filtering':
            
            for f in range(len(self.source_animation)):
                a=(self.source_animation[f]>filter_ratio).astype(int)
                b=(((a[:, :, 0]==1 ) & (a[:, :, 1]==1) & (a[:, :, 2]==1))!=True).astype(float)
                masks.append(b)
            
        return masks
        
        
    def synthesize_object(self, masks, target_sizes, target_poses):
        # 물체가 동영상 외부로 나갈 위험성이 있을 시.
        for pad in range(30): # 30 : try to 30. 
            try:
                video=copy.deepcopy(self.vi_result)
                for f in range(len(video)):
                    w,h=target_sizes[f] # 새로 만든 animation의 크기
                    px, py=target_poses[f] # 새로 
                    cx, cy=self.origin_video_pos_512[f] # 합성은 512에 한다.
                    l, u=int(w/2)-pad, int(h/2)-pad
                    
                    # generated object의 위치를 bounding box로 받게한다면?
                    # 그러면, 아래의 add_height 등과 위의 asepct_ratio 등을 한 번에 처리 가능.
                    # 높, 낮이 조절 (양수 : 위)
                    add_height=0

                    # 좌, 우 조절 (양수 : 오른쪽)
                    add_width=0

                    # 굳이 int를 써야할까..
                    video_part=video[f][cy-u-add_height: cy+u-add_height, cx-l+add_width:cx+l+add_width]
                #     print(cy-u-2*add_height, cy+u-2*add_height)
                    ani_part=self.source_animation[f][py-u:py+u, px-l:px+l]
                    ani_part_masking=masks[f][py-u:py+u, px-l:px+l]
                    # remove

                    remover=video_part*np.broadcast_to(ani_part_masking[..., np.newaxis], ani_part_masking.shape+(3,))
                    video_part=video_part*(remover==0)

                    # sum
                    obj=mask_image(ani_part, ani_part_masking)

                    obj[obj==1]=0
                    video_part=video_part+obj

                    video[f][cy-u-add_height: cy+u-add_height, cx-l+add_width:cx+l+add_width]=video_part
                break # pad 관련 오류가 없을 시
            except Exception as e:
                if pad==29:
                    print(e)
                    raise NameError('object went off the video. adjust add_height or add_width')
                
                continue # plus 1 to pad
        
        return video
    

            

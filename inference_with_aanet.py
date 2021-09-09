import glob
from vos.test import *

import os, sys
import os.path
import numpy as np
import cv2
import pickle

from vos.vos_models.custom import Custom

from vi.model import generate_model
from vi.vi_inference import VIInference

from AANet.aa_inference import *

def assign_siammask(args, cfg, device):     
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume_aanet), 'Please download {} first.'.format(args.resume_aanet)
        siammask = load_pretrain(siammask, args.resume)
    siammask.eval().to(device)
    
    return siammask
    
def track_mask(ims, bbox, AAInf, siammask, cfg, device, method='origin video'):
    """method : origin video(1), source image(2), driving video(3)"""
    
    x,y,w,h=bbox
    toc=0
    if method=='driving video':
        target_poses=[]
        target_sizes=[]
        ims=[(im*255).astype(int) for im in ims] # SIAM : 0~255, AANet: 0~1
    
    for f, im in enumerate(ims):        

        tic = cv2.getTickCount()
        
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker

        # 첫번째 frame부터 tracking
        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)
        mask = state['mask'] > state['p'].seg_thr

        if method=='origin video':
            AAInf.origin_video_pos.append(state['target_pos'].astype(int))
            AAInf.origin_video_sz.append(state['target_sz'].astype(int))
            AAInf.origin_video_mask.append(mask.astype(int))
        elif method=='source image':
            AAInf.source_image_mask=mask.astype(int)
            AAInf.source_image=mask_image(AAInf.source_image, AAInf.source_image_mask)
        elif method=='driving video':
            target_sizes.append(state['target_sz'].astype(int))
            target_poses.append(state['target_pos'].astype(int))
            
        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    fps = f / toc

    print('track_mask({}) Time: {:02.1f}s Speed: {:3.1f}fps'.format(method, toc, fps))
          
    if method=='driving video':
        return target_sizes, target_poses
          
def aanet_main(AAInf, vi_result, args):
    
    #Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    # Setup Model
    cfg = load_config(args)

    
    AAInf.vi_result=vi_result
    
    # save original video
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]
    AAInf.origin_video=[(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255) for img in ims]
    
    # bounding box for original bounding box
    AAInf.origin_video_bbox=resize_bbox(AAInf.origin_video_bbox_512, 
                                            (512,512), (AAInf.origin_video[0]).shape[:2][::-1])
    
    ### track origin video ### 
    
    # siam mask 할당.
    siammask=assign_siammask(args, cfg, device)
    
    # bbox of object of origin video
    bbox=AAInf.origin_video_bbox # 원본 비디오에 대한 bbox1
    
    # track origin video 
    track_mask(ims, bbox, AAInf, siammask, cfg, device, method='origin video')

    ### crop video to (384, 384) ###
    AAInf.driving_video_384, AAInf.driving_video_384_mask, _=AAInf.generate_driving_video_384(
        AAInf.origin_video, AAInf.origin_video_mask, AAInf.origin_video_pos,
                            AAInf.origin_video_sz)

    ### track source image ### 
    siammask = assign_siammask(args, cfg, device)

    
    bbox=AAInf.source_image_bbox # or [160, 70, 280, 680] for 'AANet/sample/test.png' 

    AAInf.source_image_bbox=bbox ### 

    track_mask([AAInf.source_image], bbox, AAInf, siammask,cfg,device, method='source image')
    
    ### refine source image and driving video to (384, 384) shape 
    if AAInf.driving_video_384:
        AAInf.source_image_384, AAInf.driving_video_384=AAInf.resize_for_generating_animation(
  AAInf.source_image, AAInf.driving_video_384, AAInf.source_image_bbox) 
    else:
        raise NameError('use driving_video_384')
    
    ### generate animation ###
    tic=cv2.getTickCount()
    AAInf.source_animation=AAInf.generate_animation(
        AAInf.source_image_384, AAInf.driving_video_384, args.aanet_ani_mode)

    toc=cv2.getTickCount()-tic
    toc /= cv2.getTickFrequency()
    fps = len(AAInf.source_animation) / toc
    print('generating animation Time: {:02.1f}s Speed: {:3.1f}fps)'.format(toc, fps))

    ### track driving video(animation video) ### 

    siammask = assign_siammask(args, cfg, device)

    bbox=AAInf.source_image_384_bbox

    target_sizes, target_poses= track_mask(AAInf.source_animation, bbox, AAInf, siammask,cfg, device, method='driving video')

    ### decouple object from background ###
    masks=AAInf.decouple_background(AAInf.source_animation)
    
    ### synthesize object to inpainting video ### 
    video=AAInf.synthesize_object(masks, target_sizes, target_poses)
    # 비디오 합성 (0710 메서드화.)  
    return video

    
        

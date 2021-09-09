import glob
from vos.test import *

import os, sys
import os.path
import numpy as np
import cv2
import pickle
import imageio
from skimage import img_as_ubyte

from vi.model import generate_model
from vi.vi_inference import VIInference

from vos.vos_models.custom import Custom

from AANet.aa_inference import AAInference
from inference_with_aanet import aanet_main

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_inference.json',
                    help='hyper-parameter of SiamMask and VINet in json format')
parser.add_argument('--base_path', default='./vos/data/tennis', help='datasets')
parser.add_argument('--save_path', default='./results', help='save path for modified video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')

# AANet args
parser.add_argument('--using_aanet', default=False, help='if wanna use animation generating', type=str2bool)
parser.add_argument('--resume_aanet', default='SiamMask_DAVIS.pth', help='siammask parameters for aanet')
parser.add_argument('--aanet_source_image_path', default='./AANet/sample/test.png' ,help='source image for aanet' )
parser.add_argument('--aanet_config_path', default='./AANet/config/4dataset384.yaml', help='aanet_model.yaml')
parser.add_argument('--aanet_model_path', default='./AANet/checkpoints/4dataset384.pth', help='aanet_model.pth')
parser.add_argument('--aanet_ani_mode', default='relative', help='ani_mode : relative or avd')

args = parser.parse_args()


if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    vinet, _ = generate_model(cfg['opt'])
    vinet.eval()
    inf = VIInference(vinet)
   
    if args.using_aanet:
        AAInf=AAInference(args.aanet_config_path, args.aanet_model_path)
        
    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    # VI쪽으로 넘겨주기 위해 이미지 사이즈 변환 (w,h) = (512,512)
    ims = [cv2.resize(cv2.imread(imf), (512,512)) for imf in img_files]
    
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()
    
    # x, y, w, h = 170, 110, 130, 290
    
    if args.using_aanet:
        AAInf.origin_video_bbox_512=[x,y,w,h]
        AAInf.source_image=imageio.imread(args.aanet_source_image_path)
        
        cv2.namedWindow("AANet", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("AANet", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            AAInf.source_image_bbox = cv2.selectROI('AANet', AAInf.source_image, False, False)
        except:
            exit()
        
        AAInf.origin_video_512=[(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255) for img in ims]
    
    toc = 0

    for f, im in enumerate(ims):        
        # VI 쪽으로 이미지 넘겨주기 위해 저장하는 코드
        im_name = 'images/' + str(f).zfill(5) + '.jpg'
        cv2.imwrite(im_name, im)

        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            
        # 첫번째 frame부터 tracking
        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        if args.using_aanet:
            AAInf.origin_video_pos_512.append(state['target_pos'].astype(int))
            AAInf.origin_video_mask_512.append(mask.astype(int))
            
        inf.inference(im, mask)
        
        toc += cv2.getTickCount() - tic

    
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    inf.to_video("test", args.save_path)
    
    
    # AANet
    # input : inpainting video, args (단, bbox들은 우선 임의로 기재해놓았음, WebDemo.py에서는 
    
    if args.using_aanet:
        # 위에서 vinet이 생성한 inpainted video 가져오기.
        vi_result=[f/255 for f in imageio.mimread(os.path.join(args.save_path,'test.mp4'))]
        
        video=aanet_main(AAInf, vi_result, args)
                         
        imageio.mimsave(os.path.join(args.save_path, 'test_aanet.mp4'), [img_as_ubyte(frame) for frame in video])
        print(f'saved to : {args.save_path}/test_aanet.mp4')
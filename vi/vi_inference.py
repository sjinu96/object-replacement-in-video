import os, sys
import os.path
import torch
import numpy as np
import cv2  

from torch.utils import data
from torch.autograd import Variable
from .model import generate_model
import time

import subprocess as sp
import pickle


def createVideoClip(clip, folder, name, size=[512,512]):
    out = cv2.VideoWriter(folder+'/'+name,cv2.VideoWriter_fourcc(*'mp4v'), 15, (512,512))
            
    for i in range(len(clip)):
        out.write(cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB))
    out.release()

    print("video creation end")

def to_img(x):
    tmp = (x[0,:,0,:,:].cpu().data.numpy().transpose((1,2,0))+1)/2
    tmp = np.clip(tmp,0,1)*255.
    return tmp.astype(np.uint8)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class VIInference:

    def __init__(self, vi_model):
        self.vi_model = vi_model
        self.imgs = []
        self.masks = []
        self.masked_imgs = []
        self.inpainted_imgs = []
        self.t = 0
        self.ts = 3
        self.prev_mask_ = None
        self.prev_outputs = None
        self.lstm_state = None

    def inference(self, img, mask): 
        # shape of the img expected to be (3, 512, 512)
        # shape of the mask expected to be (1, 512, 512)

        """
        Change input for siammask -> VINet
        """
        img = torch.tensor(img, dtype = torch.float32)
        img /= 255.
        mask = torch.tensor(mask, dtype = torch.float32)
        img = img.permute(2,0,1)
        mask = mask.unsqueeze(0)
        

        img = 2. * img - 1
        inverse_mask = 1 - mask
        masked_img = img.clone() * inverse_mask

        img = to_var(img)
        mask = to_var(mask)
        masked_img = to_var(masked_img)

        self.imgs.append(img)
        self.masks.append(mask)
        self.masked_imgs.append(masked_img)

        masked_inputs_ = []
        masks_ = []

        if len(self.imgs) >= self.ts*2+1 and len(self.masks) >= self.ts*2+1 and len(self.masked_imgs) >= self.ts*2+1:
            if self.t < 2 * self.ts:
                masked_inputs_.append(self.masked_imgs[abs(self.t-2*self.ts)])
                masked_inputs_.append(self.masked_imgs[abs(self.t-1*self.ts)])
                masked_inputs_.append(self.masked_imgs[self.t])
                masked_inputs_.append(self.masked_imgs[self.t+1*self.ts])
                masked_inputs_.append(self.masked_imgs[self.t+2*self.ts])
                masks_.append(self.masks[abs(self.t-2*self.ts)])
                masks_.append(self.masks[abs(self.t-1*self.ts)])
                masks_.append(self.masks[self.t])
                masks_.append(self.masks[self.t+1*self.ts])
                masks_.append(self.masks[self.t+2*self.ts])
            else:
                masked_inputs_.append(self.masked_imgs[self.t-2*self.ts])
                masked_inputs_.append(self.masked_imgs[self.t-1*self.ts])
                masked_inputs_.append(self.masked_imgs[self.t])
                masked_inputs_.append(self.masked_imgs[self.t+1*self.ts])
                masked_inputs_.append(self.masked_imgs[self.t+2*self.ts])
                masks_.append(self.masks[self.t-2*self.ts])
                masks_.append(self.masks[self.t-1*self.ts])
                masks_.append(self.masks[self.t])
                masks_.append(self.masks[self.t+1*self.ts])
                masks_.append(self.masks[self.t+2*self.ts]) 
            
            masked_inputs_ = torch.stack(masked_inputs_).permute(1,0,2,3).unsqueeze(0)
            masks_ = torch.stack(masks_).permute(1,0,2,3).unsqueeze(0)

            prev_mask = masks_[:, :, 2] if self.t == 0 else self.prev_mask_
            prev_ones = to_var(torch.ones(prev_mask.size()))

            if self.t == 0:
                prev_feed = torch.cat([masked_inputs_[:,:,2,:,:], prev_ones, prev_ones*prev_mask], dim = 1)
            else:
                prev_feed = torch.cat([self.prev_outputs.detach().squeeze(2), prev_ones, prev_ones*prev_mask], dim = 1)
            
            outputs, _, _, _, _ = self.vi_model(masked_inputs_, masks_, self.lstm_state, prev_feed, self.t)
            
            self.prev_outputs = outputs
            self.prev_mask_ = masks_[:,:,2]*0.5
            
            inpainted = to_img(outputs)
            self.inpainted_imgs.append(inpainted[:,:,::-1])
            self.t += 1
    
    def to_video(self, video_name, video_path):
        final_clip = np.stack(self.inpainted_imgs)
        createVideoClip(final_clip, video_path, '%s.mp4'%video_name)

if __name__=='__main__':
    from .davis import DAVIS

    class Object():
        pass
    opt = Object()
    opt.crop_size = 512
    opt.double_size = True if opt.crop_size == 512 else False

    opt.search_range = 4 # fixed as 4: search range for flow subnetworks
    opt.pretrain_path = 'results/vinet_agg_rec/save_agg_rec_512.pth'
    opt.result_path = 'results/vinet_agg_rec'

    opt.model = 'vinet_final'
    opt.batch_norm = False
    opt.no_cuda = False # use GPU
    opt.no_train = True
    opt.test = True
    opt.t_stride = 3
    opt.loss_on_raw = False
    opt.prev_warp = True
    opt.save_image = True
    opt.save_video = True

    model, _ = generate_model(opt)
    model.eval()
    folder_name = 'davis_%d'%(int(opt.crop_size))
    DAVIS_ROOT = './DAVIS_demo'
    DTset = DAVIS(DAVIS_ROOT, imset='2016/demo_davis.txt', size=(opt.crop_size, opt.crop_size))
    DTloader = data.DataLoader(DTset, batch_size=1, shuffle=False, num_workers=1)
    for seq, (inputs, masks, info) in enumerate(DTloader):
        break

    inf = VIInference(model)
    for i in range(80):
        img = inputs[0, :, i]
        mask = masks[0, :, i]

        
        inf.inference(img, mask)
    inf.to_video("tmptmp", "./DAVIS_demo")
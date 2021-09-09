import os, sys
import os.path
import torch
import numpy as np
import cv2  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from davis import DAVIS
from model import generate_model
import time
from pytorch_ssim import SSIM
from resample2d import Resample2d
import flownet2

import subprocess as sp
import pickle

class Object0(object):
    pass
args = Object0()
args.rgb_max = 1.0
args.fp16 = False
# FlowNet = flownet2.FlowNet2(args, requires_grad=False)
FlowNet = flownet2.FlowNet2(args)
model_filename = os.path.join(
    "pretrained_models", "FlowNet2_checkpoint.pth.tar")
checkpoint = torch.load(model_filename)
FlowNet.load_state_dict(checkpoint['state_dict'])
FlowNet = FlowNet.cuda()
""" Submodules """
flow_warping = Resample2d().cuda()
downsampler = nn.AvgPool2d((2, 2), stride=2).cuda()

class Object():
    pass
opt = Object()
opt.crop_size = 512
opt.double_size = True if opt.crop_size == 512 else False
# DAVIS dataloader
DAVIS_ROOT = './DAVIS_demo'
DTset = DAVIS(DAVIS_ROOT, imset='2016/demo_davis.txt', 
              size=(opt.crop_size, opt.crop_size))
DTloader = data.DataLoader(DTset, batch_size=1, shuffle=False, num_workers=1)

opt.search_range = 4 # fixed as 4: search range for flow subnetworks
opt.pretrain_path = 'results/vinet_agg_rec/save_agg_rec_512.pth'
opt.result_path = 'results/vinet_agg_rec'

opt.model = 'vinet_final'
opt.batch_norm = False
opt.no_cuda = False # use GPU
opt.no_train = False
opt.test = False
opt.t_stride = 3
opt.loss_on_raw = False
opt.prev_warp = True
opt.save_image = False
opt.save_video = False
opt.no_lstm = False
opt.w_ST = True
opt.w_LT = True

def norm(t):
    return torch.sum(t*t, dim=1, keepdim=True) 

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

model, _ = generate_model(opt)
print('Number of model parameters: {}'.format(
      sum([p.data.nelement() for p in model.parameters()])))

model.train()
ts = opt.t_stride
folder_name = 'davis_%d'%(int(opt.crop_size))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion_L1 = torch.nn.L1Loss()
criterion_ssim = SSIM(window_size = 11)

for seq, (inputs, masks, info) in enumerate(DTloader):

    bs = inputs.size(0)
    num_frames = inputs.size(2)
    seq_name = info['name'][0]
    midx = 2

    save_path = os.path.join(opt.result_path, folder_name, seq_name)
    if not os.path.exists(save_path) and opt.save_image:
        os.makedirs(save_path)

    inputs = 2.*inputs - 1
    inverse_masks = 1-masks
    masked_inputs = inputs.clone()*inverse_masks

    masks = to_var(masks)
    masked_inputs = to_var(masked_inputs)
    inputs = to_var(inputs)

    frame_i, frame_mi, frame_m = [], [], []

    for t in range(num_frames):
        masked_inputs_ = []
        masks_ = []       
        inputs_ = [] 

        if t < 2 * ts:
            masked_inputs_.append(masked_inputs[0,:,abs(t-2*ts)])
            masked_inputs_.append(masked_inputs[0,:,abs(t-1*ts)])
            masked_inputs_.append(masked_inputs[0,:,t])
            masked_inputs_.append(masked_inputs[0,:,t+1*ts])
            masked_inputs_.append(masked_inputs[0,:,t+2*ts])
            masks_.append(masks[0,:,abs(t-2*ts)])
            masks_.append(masks[0,:,abs(t-1*ts)])
            masks_.append(masks[0,:,t])
            masks_.append(masks[0,:,t+1*ts])
            masks_.append(masks[0,:,t+2*ts])
            inputs_.append(inputs[0,:,abs(t-2*ts)])
            inputs_.append(inputs[0,:,abs(t-1*ts)])
            inputs_.append(inputs[0,:,abs(t)])
            inputs_.append(inputs[0,:,abs(t+1*ts)])
            inputs_.append(inputs[0,:,abs(t+2*ts)])
        elif t > num_frames - 2 * ts - 1:
            masked_inputs_.append(masked_inputs[0,:,t-2*ts])
            masked_inputs_.append(masked_inputs[0,:,t-1*ts])
            masked_inputs_.append(masked_inputs[0,:,t])
            masked_inputs_.append(masked_inputs[0,:,-1 -abs(num_frames-1-t - 1*ts)])
            masked_inputs_.append(masked_inputs[0,:,-1 -abs(num_frames-1-t - 2*ts)])
            masks_.append(masks[0,:,t-2*ts])
            masks_.append(masks[0,:,t-1*ts])
            masks_.append(masks[0,:,t])
            masks_.append(masks[0,:,-1 -abs(num_frames-1-t - 1*ts)])
            masks_.append(masks[0,:,-1 -abs(num_frames-1-t - 2*ts)])   
            inputs_.append(inputs[0,:,t-2*ts])
            inputs_.append(inputs[0,:,t-1*ts])
            inputs_.append(inputs[0,:,t])
            inputs_.append(inputs[0,:,-1 -abs(num_frames-1-t - 1*ts)])
            inputs_.append(inputs[0,:,-1 -abs(num_frames-1-t - 2*ts)])
        else:
            masked_inputs_.append(masked_inputs[0,:,t-2*ts])
            masked_inputs_.append(masked_inputs[0,:,t-1*ts])
            masked_inputs_.append(masked_inputs[0,:,t])
            masked_inputs_.append(masked_inputs[0,:,t+1*ts])
            masked_inputs_.append(masked_inputs[0,:,t+2*ts])
            masks_.append(masks[0,:,t-2*ts])
            masks_.append(masks[0,:,t-1*ts])
            masks_.append(masks[0,:,t])
            masks_.append(masks[0,:,t+1*ts])
            masks_.append(masks[0,:,t+2*ts])
            inputs_.append(inputs[0,:,t-2*ts])
            inputs_.append(inputs[0,:,t-1*ts])
            inputs_.append(inputs[0,:,t])
            inputs_.append(inputs[0,:,t+1*ts])
            inputs_.append(inputs[0,:,t+2*ts])            

        masked_inputs_ = torch.stack(masked_inputs_).permute(
            1,0,2,3).unsqueeze(0)
        masks_ = torch.stack(masks_).permute(1,0,2,3).unsqueeze(0)
        inputs_ = torch.stack(inputs_).permute(1,0,2,3).unsqueeze(0)

        frame_i.append(to_var(inputs_))
        frame_mi.append(to_var(masked_inputs_))
        frame_m.append(to_var(masks_))
    
    optimizer.zero_grad()
    lstm_state = None
    ST_loss, LT_loss = 0, 0
    RECON_loss, HOLE_loss = 0, 0
    flow_loss = 0


    ### forward
    prev_mask = frame_m[0][:,:,midx,:,:]
    prev_ones = to_var(torch.ones(prev_mask.size()))
    prev_feed = torch.cat([frame_mi[0][:,:,midx,:,:], prev_ones, prev_ones*prev_mask], dim = 1)

    frame_o1, _, lstm_state, _, occs = model(frame_mi[0], frame_m[0], lstm_state, prev_feed)
    frame_o1 = frame_o1.squeeze(2)


    RECON_loss += 1*criterion_L1(frame_o1, frame_i[0][:,:,midx,:,:]) - criterion_ssim(frame_o1, frame_i[0][:, :, midx, :, :])
    HOLE_loss += 5*criterion_L1(
            frame_o1*frame_m[0][:,:,midx,:,:].expand_as(frame_o1), 
            frame_i[0][:,:,midx,:,:]*frame_m[0][:,:,midx,:,:].expand_as(frame_o1)
            )
  
    frame_o = []
    frame_o.append(frame_o1)

    for tt in range(1, num_frames):
        frame_i1, frame_m1 = frame_i[tt-1], frame_m[tt-1]
        frame_mi2 = frame_mi[tt]
        frame_i2, frame_m2 = frame_i[tt], frame_m[tt]
        frame_o1 = frame_o1.detach() if tt == 1 else frame_o2.detach()

        prev_mask = to_var(torch.zeros(frame_m2[:,:,midx,:,:].size()))
        prev_ones = to_var(torch.ones(prev_mask.size()))
        prev_feed = torch.cat([frame_o1, prev_ones, prev_ones*prev_mask], dim = 1)

        frame_o2, _, lstm_state, occs, flow6_256 = model(frame_mi2, frame_m2, lstm_state, prev_feed, 1)

        if opt.loss_on_raw:
            frame_o2_raw = frame_o2[1].squeeze(2)
            frame_o2 = frame_o2[0]
        frame_o2 = frame_o2.squeeze(2)

        lstm_state = None if opt.no_lstm else repackage_hidden(lstm_state)
        frame_o.append(frame_o2)
      
        RECON_loss += criterion_L1(frame_o2, frame_i2[:,:,midx,:,:]) - criterion_ssim(frame_o2, frame_i2[:,:,midx,:,:])
        HOLE_loss += 5*criterion_L1(
            frame_o2 * frame_m2[:,:,midx,:,:].expand_as(frame_o2), 
            frame_i2[:,:,midx,:,:] * frame_m2[:,:,midx,:,:].expand_as(frame_o2))
        
        if opt.w_ST:
            flow_i21 = FlowNet(torch.stack([frame_i2[:,:,midx,:,:], frame_i1[:,:,midx,:,:]], dim = 2)) # flow of tt and tt-1
            warp_i1 = flow_warping(frame_i1[:,:,midx,:,:], flow_i21) 
            warp_o1 = flow_warping(frame_o1, flow_i21)
            noc_mask2 = torch.exp( -50. * torch.sum(frame_i2[:,:,midx,:,:] - warp_i1, dim=1).pow(2) ).unsqueeze(1)
            ST_loss += criterion_L1(frame_o2 * noc_mask2, warp_i1 * noc_mask2)

            conf = (norm(frame_i2[:,:,midx,:,:] - warp_i1) < 0.02).float()
            flow_loss = criterion_L1(flow6_256 * conf, flow_i21 * conf)
            warp_i1_ = flow_warping(frame_i1[:,:,midx,:,:], flow6_256)
            flow_loss += criterion_L1(warp_i1_ * conf, frame_i2[:,:,midx,:,:] * conf)
            warp_o1_ = flow_warping(frame_o1, flow6_256)
            flow_loss += criterion_L1(frame_o2 * conf, warp_o1_.detach() * conf)

    if opt.w_LT:
        t1 = 0
        for t2 in range(t1 + 2, num_frames):
            frame_i1, frame_i2 = frame_i[t1], frame_i[t2]
            frame_o1 = frame_o[t1].detach()
            frame_o1.requires_grad = False
            frame_o2 = frame_o[t2]

            flow_i21 = FlowNet(torch.stack([frame_i2[:,:,midx,:,:], frame_i1[:,:,midx,:,:]], dim = 2))
            warp_i1 = flow_warping(frame_i1[:,:,midx,:,:], flow_i21)
            warp_o1 = flow_warping(frame_o1, flow_i21)
            noc_mask2 = torch.exp(-50. * torch.sum(frame_i2[:,:,midx,:,:] - warp_i1, dim=1).pow(2) ).unsqueeze(1)
            LT_loss += criterion_L1(frame_o2 * noc_mask2, warp_i1 * noc_mask2)
    
    overall_loss = (RECON_loss + HOLE_loss + opt.w_ST * ST_loss  + opt.w_LT * LT_loss + opt.w_FLOW * flow_loss)
    print(overall_loss)
    overall_loss.backward()
    optimizer.step()
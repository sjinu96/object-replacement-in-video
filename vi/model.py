import torch
from torch import nn
from models import vinet


def generate_model(opt):
    if type(opt) == dict:
        class Object():
            pass
        opt_ = Object()
        opt_.crop_size = opt['crop_size']
        opt_.double_size = opt['double_size']

        opt_.search_range = opt['search_range'] # fixed as 4: search range for flow subnetworks
        opt_.pretrain_path = opt['pretrain_path']
        opt_.result_path = opt['result_path']

        opt_.model = opt['model']
        opt_.batch_norm = opt['batch_norm']
        opt_.no_cuda = opt['no_cuda'] # use GPU
        opt_.no_train = opt['no_train']
        opt_.test = opt['test']
        opt_.t_stride = opt['t_stride']
        opt_.loss_on_raw = opt['loss_on_raw']
        opt_.prev_warp = opt['prev_warp']
        opt_.save_image = opt['save_image']
        opt_.save_video = opt['save_video']
        opt = opt_

    #try: 
    #    assert(opt.model == 'vinet_final')
    #    model = vinet.VINet_final(opt=opt)
    #except:
    #    print('Model name should be: vinet_final')
    model = vinet.VINet_final(opt = opt)
    assert(opt.no_cuda is False)
    model = model.cuda()
    model = nn.DataParallel(model)
    loaded, empty = 0,0
    if opt.pretrain_path:
        print('Loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)

        child_dict = model.state_dict()
        parent_list = pretrain['state_dict'].keys()
        parent_dict = {}
        for chi,_ in child_dict.items():
            if chi in parent_list:
                parent_dict[chi] = pretrain['state_dict'][chi]
                #print('Loaded: ',chi)
                loaded += 1
            else:
                #print('Empty:',chi)
                empty += 1
        print('Loaded: %d/%d params'%(loaded, loaded+empty))
        child_dict.update(parent_dict)
        model.load_state_dict(child_dict)   

    return model, model.parameters()

# ---------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# ---------------------------------------------------------
import glob
from test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='./data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from vos_models.custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    # VI쪽으로 넘겨주기 위해 이미지 사이즈 변환 (w,h) = (512,512)
    ims = [cv2.resize(cv2.imread(imf), (512,512)) for imf in img_files]

    # Select ROI
    # GUI 문제 (X server)

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()

    # bounding box coordinates입력
    # x, y, w, h = 300, 110, 165, 250

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

            # VI쪽으로 마스크 넘겨주기 위해 저장하는 코드
            mask = np.array(mask*255, dtype='uint8')  # 이미지로 저장하기 위해 형변환
            mask_name = 'masks/'+str(f).zfill(5) + '.jpg'
            cv2.imwrite(mask_name, mask)

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            name = 'output/' + str(f).zfill(5) + '.jpg'
            cv2.imwrite(name, im)
            key = cv2.waitKey(1)
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            # VI쪽으로 마스크 넘겨주기 위해 저장하는 코드
            mask = np.array(mask*255, dtype='uint8')  # 이미지로 저장하기 위해 형변환
            mask_name = 'masks/'+str(f).zfill(5) + '.jpg'
            cv2.imwrite(mask_name, mask)

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            """
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key > 0:
                break
            """
            name = 'output/' + str(f).zfill(5) + '.jpg'
            cv2.imwrite(name, im)
            key = cv2.waitKey(1)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
import os
import cv2
import logging
import numpy as np
from torch.utils.data import Dataset
# from projects.BiSeNet_zzb.liusheng.bisenetV2 import transform_cv2 as T
from bisenetV2 import transform_cv2 as T


class TransformationTrain(object):
    def __init__(self, logger: logging.Logger, configs: dict):
        trans = []
        try:
            if configs['train_process']['use_resize']:
                trans.append(T.Resized(configs['train_process']['new_size']))
            if configs['train_process']['use_random_hor_flip']:
                trans.append(T.RandomHorizontalFlip())
            if configs['train_process']['use_img_hsvJitter']:
                trans.append(T.ColorJitter(
                    brightness=configs['train_process']['jitter_range'][0],
                    contrast=configs['train_process']['jitter_range'][1],
                    saturation=configs['train_process']['jitter_range'][2]))
            if configs['train_process']['use_img_rotation']:
                trans.append(T.RandomRotation(configs['train_process']['rotate_degree']))
            if configs['train_process']['use_overlap']:
                print(f"print in trans: {configs['train_process']['margin']}")
                trans.append(T.RandomOverlap(configs['train_process']['margin']))

            if configs['train_process']['use_grey_img']:
                trans.append(T.ImgTransformGray())
        except Exception as msg_e:
            logger.error('\n' + msg_e)
            exit(1)
        self.trans_func = T.Compose(trans)

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):
    def __init__(self, logger: logging.Logger, configs: dict):
        trans = []
        try:
            if configs['train_process']['use_resize']:
                trans.append(T.Resized(configs['train_process']['new_size']))
            if configs['train_process']['use_grey_img']:
                trans.append(T.ImgTransformGray())
        except Exception as msg_e:
            logger.error('\n' + msg_e)
            exit(1)
        self.trans_func = T.Compose(trans)

    def __call__(self, im_lb):
        # trans.append(T.Resized(configs['train_process']['new_size']))
        # im, lb = im_lb['im'], im_lb['lb']
        # return dict(im=im, lb=lb)
        im_lb = self.trans_func(im_lb)
        return im_lb


class BaseDataset(Dataset):
    def __init__(self, dataset_root, trans_func=None, phrase='train'):
        super(BaseDataset, self).__init__()
        assert phrase in ('train', 'val', 'test')

        if phrase == 'train':
            ori_phrase = 'TR-Image'
            mask_phrase = 'TR-Mask0'
        else:
            ori_phrase = 'VL-Image'
            mask_phrase = 'VL-Mask0'

        self.phrase = phrase
        self.trans_func = trans_func

        self.lb_map = None

        ori_imgs = os.listdir(os.path.join(dataset_root, ori_phrase))
        ori_imgs = list(filter(lambda x: x.endswith(('.jpg', '.png')), ori_imgs))
        ori_imgs.sort(reverse=False)
        self.img_paths = list(map(lambda x: os.path.join(dataset_root, ori_phrase, x), ori_imgs))

        masks = os.listdir(os.path.join(dataset_root, mask_phrase))
        masks = list(filter(lambda x: x.endswith(('.jpg', '.png')), masks))
        masks.sort(reverse=False)
        self.lb_paths = list(map(lambda x: os.path.join(dataset_root, mask_phrase, x), masks))
        # with open(ann_path, 'r') as fr:
        #     pairs = fr.read().splitlines()
        # self.img_paths, self.lb_paths = [], []
        # for pair in pairs:
        #     imgpth, lbpth = pair.split(',')
        #     self.img_paths.append(os.path.join(dataset_root, imgpth))
        #     self.lb_paths.append(os.path.join(dataset_root, lbpth))
        assert len(self.img_paths) == len(self.lb_paths)

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        # print('impth', impth)
        # print('lbpth', lbpth)
        # BGR -> RGB
        img, label = cv2.imread(impth)[:, :, ::-1], cv2.imread(lbpth, 0)
        if self.lb_map is not None:
            label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if self.trans_func is not None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return img.detach(), label.unsqueeze(0).detach()

    def __len__(self):
        return len(self.img_paths)


# wool
# labels_info = [
#     {"name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 0},
#     {"name": "yixian", "ignoreInEval": True, "id": 1, "color": [255, 0, 0], "trainId": 1},  # 嵌饰板
#     {"name": "shenhuangmao", "ignoreInEval": True, "id": 2, "color": [0, 255, 0], "trainId": 2},  # 扶手
# ]


class sjhtCommon(BaseDataset):
    def __init__(self, dataset_root, class_data, trans_func=None, phrase='train'):
        super(sjhtCommon, self).__init__(
            dataset_root, trans_func, phrase)
        self.n_cats = len(class_data)  # numclass
        self.lb_ignore = 255
        self.lb_map = np.arange(self.n_cats).astype(np.uint8)
        # self.lb_map = None
        # for el in labels_info:
        #     self.lb_map[el['id']] = el['trainId']
        for idx, cls in enumerate(class_data):
            # class name is temporary useless
            self.lb_map[idx] = idx

        # city, rgb
        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115)
        )

#!/usr/bin/python
# -*- encoding: utf-8 -*-


import random
import math

import numpy as np
import cv2
import torch


class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''

    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im, lb=lb)
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh + crop_h, sw:sw + crop_w, :].copy(),
            lb=lb[sh:sh + crop_h, sw:sw + crop_w].copy()
        )


class Resized(object):
    '''
        size should be a tuple of (H, W)
    '''

    def __init__(self, size=(384, 384)):
        self.size = size

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        # assert im.shape[:2] == lb.shape[:2]
        resize_h, resize_w = self.size[0], self.size[1]
        im = cv2.resize(im, (resize_w, resize_h))
        if lb is not None:
            lb = cv2.resize(lb, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        return dict(
            im=im,
            lb=lb
        )


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[:, ::-1, :],
            lb=lb[:, ::-1],
        )


class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            im = self.adj_brightness(im, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            im = self.adj_contrast(im, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            im = self.adj_saturation(im, rate)
        return dict(im=im, lb=lb, )

    def adj_saturation(self, im, rate):
        M = np.float32([
            [1 + 2 * rate, 1 - rate, 1 - rate],
            [1 - rate, 1 + 2 * rate, 1 - rate],
            [1 - rate, 1 - rate, 1 + 2 * rate]
        ])
        shape = im.shape
        im = np.matmul(im.reshape(-1, 3), M).reshape(shape) / 3
        im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    def adj_brightness(self, im, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

    def adj_contrast(self, im, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]


class hsvJitter(object):
    def __init__(self, class_num):
        # 暂时只支持1类+background
        self.class_num = class_num

    def random_hue(self, img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            # img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
            img_hsv[:, :, 0] += hue_delta
        return img_hsv

    def random_saturation(self, img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            # img_hsv[:, :, 1] *= sat_mult
            img_hsv[:, :, 1] += sat_mult
        return img_hsv

    def random_value(self, img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            # img_hsv[:, :, 2] *= val_mult
            img_hsv[:, :, 2] += val_mult
        return img_hsv

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        if self.class_num == 2:
            img_cv = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            img_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
            if np.random.randint(0, 2):
                img_hsv = self.random_value(img_hsv, 255)
                img_hsv = self.random_saturation(img_hsv, 255)
                img_hsv = self.random_hue(img_hsv, 255)
            else:
                img_hsv = self.random_saturation(img_hsv, 255)
                img_hsv = self.random_hue(img_hsv, 255)
                img_hsv = self.random_value(img_hsv, 255)

            img_hsv = np.clip(img_hsv, 0, 255)
            img_hsv_cv = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            img_mask_b = np.where(lb == 1, 0, 1).astype(np.uint8)
            img_mask_b_3 = np.stack((img_mask_b,) * 3, axis=-1)
            lb_3 = np.stack((lb,) * 3, axis=-1)
            img = img_hsv_cv * lb_3 + img_mask_b_3 * img_cv
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_lb['im'] = img_rgb
        return im_lb


'''
随机旋转图片和mask
'''


class RandomRotation(object):
    def __init__(self, angle_value):
        self.angle_value = angle_value

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        angle = random.randint(-1 * self.angle_value, self.angle_value)
        w, h = im.shape[1], im.shape[0]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_change = cv2.warpAffine(im, M, (w, h), borderValue=(0, 0, 0))
        lb_mask = cv2.warpAffine(lb, M, (w, h), borderValue=0)
        return dict(
            im=img_change,
            lb=lb_mask
        )


'''
将图像拼接成由灰度图组成的三通道图片
'''


class ImgTransformGray(object):
    def __init__(self, with_gray=True):
        self.with_gray = with_gray

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        # assert im.shape[:2] == lb.shape[:2]
        if self.with_gray:
            im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            stacked_img = np.stack((im_gray,) * 3, axis=-1)
            im_lb['im'] = stacked_img
        return im_lb


def iou(rec1, rec2):
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        return 1


"""
随机在线增强
"""


class RandomOverlap(object):
    def __init__(self, margin):
        self.margin = margin

    def __call__(self, im_lb):
        img, mask = im_lb['im'], im_lb['lb']
        foreground = np.where(mask > 0)
        h0, w0, h1, w1 = np.min(foreground[0]), np.min(foreground[1]), np.max(foreground[0]), np.max(foreground[1])
        img_h, img_w = mask.shape
        h0, w0, h1, w1 = max(h0 - self.margin, 0), \
                         max(w0 - self.margin, 0), \
                         min(h1 + self.margin, img_h), \
                         min(w1 + self.margin, img_w)
        bbox_h, bbox_w = h1 - h0, w1 - w0
        # step3. transform bbox
        success = False

        for _ in range(3):
            offset_h = np.random.randint(0, img_h - bbox_h)
            offset_w = np.random.randint(0, img_w - bbox_w)
            _h0, _w0, _h1, _w1 = offset_h, offset_w, offset_h + bbox_h, offset_w + bbox_w
            # CV2_img 中坐标和np.ndarray 相反
            # x=w, y=h
            if iou([w0, h0, w1, h1], [_w0, _h0, _w1, _h1]) == 0:
                success = True
                break
        if not success:
            for _ in range(3):
                offset_h = np.random.randint(- bbox_h // 2, img_h - bbox_h // 2)
                offset_w = np.random.randint(- bbox_w // 2, img_w - bbox_w // 2)
                _h0, _w0, _h1, _w1 = offset_h, offset_w, offset_h + bbox_h, offset_w + bbox_w
                if iou([w0, h0, w1, h1], [_w0, _h0, _w1, _h1]) == 0:
                    _h0, _w0, _h1, _w1 = max(_h0, 0), max(_w0, 0), min(_h1, img_h), min(_w1, img_w)
                    # print(_h0, _w0, _h1, _w1)
                    # 暂时实现一个简单版
                    h1, w1 = h0 + (_h1 - _h0), w0 + (_w1 - _w0)
                    success = True
                    break
        if success:
            img[_h0:_h1, _w0:_w1, :] = img[h0:h1, w0:w1, :]
            mask[_h0:_h1, _w0:_w1] = mask[h0:h1, w0:w1]

        im_lb['im'], im_lb['lb'] = img, mask
        return im_lb


class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''

    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


if __name__ == '__main__':
    pass

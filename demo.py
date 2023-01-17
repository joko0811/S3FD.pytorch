#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--input_video', type=str, default='video/jannabi_clip.mp4',
                    help='Directory for detect result')
parser.add_argument('--save_dir', type=str, default='tmp/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/s3fd.pth', help='trained model')
parser.add_argument('--thresh', default=0.6, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh):
    #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                      fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    #image = cv2.resize(img, (640, 640))
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = torch.from_numpy(x).unsqueeze(0)
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    with torch.no_grad():
        y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            left_up, right_bottom = (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3]))
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.3f}".format(score)
            point = (int(left_up[0]), int(left_up[1] - 5))
            #cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,
            #            0.6, (0, 255, 0), 1)

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)

def detect_image(net, img_orig, thresh):
    t1 = cv2.getTickCount()

    img = cv2.cvtColor(img_orig.copy(), cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    # image = cv2.resize(img, (640, 640))
    image = cv2.resize(image, None, fx=1/8, fy=1/8)
    # print (image.shape)
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = torch.from_numpy(x).unsqueeze(0)
    if use_cuda:
        x = x.cuda()

    net.eval()
    with torch.no_grad():
        y = net(x)
    detections = y.data

    time = (cv2.getTickCount() - t1) / cv2.getTickFrequency() * 1000
    print('time:{:.2f}ms'.format(time))

    img = img_orig.copy()
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    list_bbox_tlbr = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            score = detections[0, i, j, 0].cpu().numpy()
            # left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            list_bbox_tlbr.append([pt[1], pt[0], pt[3], pt[2], float(score)])
            j += 1

    return list_bbox_tlbr

    # cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)

if __name__ == '__main__':
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_path = './img'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]

    for path in img_list:
        detect(net,path,args.thresh)
    """
    vc = cv2.VideoCapture(args.input_video)

    i = 0
    while True:
        i += 1
        img = vc.read()[1]
        if img is None:
            break
        if i%2 == 0:
            continue
        show = img.copy()

        list_bbox_tlbr = detect_image(net, img, args.thresh)

        for bbox in list_bbox_tlbr:
            t,l,b,r,conf = bbox
            cv2.rectangle(show, (int(l),int(t)), (int(r),int(b)), (0, 0, 255), 2)
            conf = "{:.2f}".format(conf)
            point = (int(l), int(t - 5))
            cv2.putText(show, conf, point, cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        # cv2.imshow('show', show)
        cv2.imwrite(f'hoge/hoge{i}.png',show)
        key = cv2.waitKey(1)
        if key == 27:
            break
    """

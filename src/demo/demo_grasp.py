from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np

import matplotlib.pyplot as plt


# set the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import _init_paths
from detectors.detector_factory import detector_factory
from opts import opts


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
        opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
              time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        # assumes the color image is named color_image_*, depth image is name depth_npy_*.npy
        if os.path.isdir(opt.demo):
            rgb_names = []
            depth_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext and "color_image" in file_name:
                    rgb_names.append(os.path.join(opt.demo, file_name))
        else:
            rgb_names = [opt.demo]
            depth_names = []

        # get the depth file name
        for rgb_file in rgb_names:
            dir_name = os.path.dirname(rgb_file)
            file_name = os.path.basename(rgb_file)
            idx = file_name[:file_name.rfind('.')]
            idx = idx[idx.rfind('_') + 1:]
            depth_file = os.path.join(dir_name, "depth_npy_"+idx+".npy")
            depth_names.append(depth_file)
        
        for rgb_name, depth_name in zip(rgb_names, depth_names):
            # load data
            img = cv2.imread(rgb_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.medianBlur(img,5)
            #cv2.imshow("Blured image", img_blur)
            #cv2.imshow("Original image", img)
            #cv2.waitKey()
            depth_raw = np.load(depth_name)
            image = img.astype(np.float32)
            image[:, :, 2] = depth_raw

            # run
            ret = detector.run(image)
            time_str = ''
    for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

class MultiPoseDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)
    if self.opt.inspect_aug:
      cv2.imshow("original_image", img)

    height, width = img.shape[0], img.shape[1]
    # the cropping center coordinate c
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    # maximum dimension s
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0


    ##############################################
    ###     Transform the input image
    ###     Including: random crop, rotate, flip, color jittering
    ###############################################
    flipped = False
    if self.split == 'train':
      # random square cropping (optional)
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        # If not random crop, apply random shift and scaling
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      # random rotation - NOTE: aug_rot is set to 0, so no rotation is applied!
      if np.random.random() < self.opt.aug_rot:
        rf = self.opt.rotate
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      # random flip
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1

    # apply the geometric augmentation
    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_res, self.opt.input_res])
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)

    # normalize the input 
    inp = (inp.astype(np.float32) / 255.)


    if self.opt.inspect_aug:
      cv2.imshow("Augmented image before color jitter", (inp*255).astype(np.uint8))

    # color augmentation
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

    if self.opt.inspect_aug:
      cv2.imshow("Augmented image", (inp*255).astype(np.uint8))

    # standardize
    inp = (inp - self.mean) / self.std

    # (H, W, 3) to (3, H, W) for pytorch
    inp = inp.transpose(2, 0, 1)


    ######################################################################
    ########## Generate and Transform the labels
    ######################################################################
    output_res = self.opt.output_res
    num_joints = self.num_joints
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_res, output_res), 
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res), 
                               dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      # The original bounding box (4, )
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(ann['category_id']) - 1
      # The keypoints for that object (N, 3): (x, y, v), where v is the visibility
      pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)


      # apply the flipping augmentation on the bbox and kpts
      if flipped:
        # flip the y coordinates
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        pts[:, 0] = width - pts[:, 0] - 1
        # for keypoints, also need to flip the kpt idx since left become right and right becomes left
        for e in self.flip_idx:
          pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
      
      # apply the square cropping on the bbox. BBox is not oriented, since no rotation applied
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      # If out of the bound, clip to inbound
      bbox = np.clip(bbox, 0, output_res - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0) or (rot != 0):
        # The Gaussian radius
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius)) 
        # Get the wh, center subpixel offset, index, reg_mask
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_res + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1

        # num_kpts == 0 means that the keypoint is not labeled in coco. Which means x, y should be zero
        num_kpts = pts[:, 2].sum()
        if num_kpts == 0:
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
          reg_mask[k] = 0

        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius)) 
        for j in range(num_joints):
          if pts[j, 2] > 0:
            # Apply all the augmentation on the keypoint transformation
            pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)

            # if keypoint is in bound after transformation:
            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
               pts[j, 1] >= 0 and pts[j, 1] < output_res:
              # fill in the kpts-center(int) offset
              kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
              # set the indicator mask of the corresponding x,y element to 1
              kps_mask[k, j * 2: j * 2 + 2] = 1

              # fill in the kpts offset, index, and mask
              pt_int = pts[j, :2].astype(np.int32)
              hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
              hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
              hp_mask[k * num_joints + j] = 1

              # NOTE: unknown. Dense_hp
              if self.opt.dense_hp:
                # must be before draw center hm gaussian
                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int, 
                               pts[j, :2] - ct_int, radius, is_offset=True)
                draw_gaussian(dense_kps_mask[j], ct_int, radius)
              
              # draw the kpt heatmap
              draw_gaussian(hm_hp[j], pt_int, hp_radius)
        
        # draw the center heatmap
        draw_gaussian(hm[cls_id], ct_int, radius)

        # gt detection. 
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1] + 
                       pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'hps': kps, 'hps_mask': kps_mask}

    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']

    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 40), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta

    if self.opt.inspect_aug:
      cv2.waitKey() 
    return ret

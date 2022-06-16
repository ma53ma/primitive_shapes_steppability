from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds

from utils.keypoints import get_ori_cls, ori_cls_2_angle


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def grasp_pose_post_process(opt, dets, c, s, h, w):
  """Post process the results to get the original-size image coords

  Args:
      dets (tensor, [B, K, D]): The network detection results. 
              detections (tensor, [B, K, D]): Stack of the 2d keypoint projections, width, and the relevant info including:\
            (1) scores - the center Point probability
            i.e. detections = [center_locations, kpt_locations, open_width, scores, ori_cls]
            B is the batch size. K is the candidate number \
            D = 2 + 2*num_kpts + 1 + 1 + ... 
      c (_type_): _description_
      s (_type_): _description_
      h (_type_): _description_
      w (_type_): _description_

  Returns:
      ret :  {1: preds}, where preds is: \
        preds (array, (K, D)), where K is the top-K results. D is the dimension number for describing the results, \
          including: center_coords, kpts_coords, open width, the scores.
          Hence: D = 2 + 2*kpts_num + 1 + 1
  """
  ret = []
  for i in range(dets.shape[0]):
    dets_filtered = filter_dets(opt, dets[i])
    centers = transform_preds(dets_filtered[:, :2].reshape(-1, 2), c[i], s[i], (w, h))
    kpts = transform_preds(dets_filtered[:, 2:10].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [centers.reshape(-1, 2), kpts.reshape(-1, 8),
        dets_filtered[:, 10:]], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def filter_dets(opt, dets):
  """_summary_

  Args:
      opt (_type_): _description_
      dets (K, D, np.array): _description_

  Returns:
      dets (K', D): The filtered results. K' <= K
  """
  K, D = dets.shape
  # In the orientation mode, filter according to the orientation
  if opt.ori_clf_mode:
    # raise NotImplementedError("Haven't tested the orientation-based filtering")
    ori_cls_pred = dets[:, 12]
    lr_kpts = dets[:, 2 : 10].reshape((K, 4, 2))
    ori_cls_kpts = get_ori_cls(lr_kpts, range_mode=0, total_cls_num=opt.ori_num)
    # print(ori_cls_pred)
    # print(ori_cls_kpts)
    # print(ori_cls_kpts == ori_cls_pred)
    mask = (ori_cls_kpts == ori_cls_pred)
    dets = dets[mask, :].reshape(-1, D)

  return dets 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from random import choice
import sys
import numpy as np

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='grasp_pose',
                             help='ctdet | ddd | multi_pose | exdet | grasp_pose')
    self.parser.add_argument('--dataset', default='coco',
                             help='coco | kitti | coco_hp | pascal')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk'
                                  '5: save the visualizations without displaying them')
    self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 

    # system
    self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    
    # model
    self.parser.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested'
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    
    # train
    self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--lr_step', type=str, default='90,120',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=140,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')

    # test
    self.parser.add_argument('--flip_test', action='store_true',
                             help='flip data augmentation.')
    self.parser.add_argument('--test_scales', type=str, default='1',
                             help='multi scale test augmentation.')
    self.parser.add_argument('--nms', action='store_true',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=100,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')

    # dataset
    self.parser.add_argument('--not_rand_crop', action='store_true',
                             help='not use the random crop data augmentation'
                                  'from CornerNet.')
    self.parser.add_argument('--fix_crop', action="store_true",
                            help="Remove the randomness in cropping by cropping at the center with fixed scale. For debug")
    self.parser.add_argument('--shift', type=float, default=0.1,
                             help='when not using random crop'
                                  'apply shift augmentation.')
    self.parser.add_argument('--scale', type=float, default=0.4,
                             help='when not using random crop'
                                  'apply scale augmentation.')
    self.parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    self.parser.add_argument('--flip', type = float, default=0.5,
                             help='probability of applying flip augmentation.')
    self.parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')
    # multi_pose
    self.parser.add_argument('--aug_rot', type=float, default=0, 
                             help='probability of applying '
                                  'rotation augmentation.')
    # ddd
    self.parser.add_argument('--aug_ddd', type=float, default=0.5,
                             help='probability of applying crop augmentation.')
    self.parser.add_argument('--rect_mask', action='store_true',
                             help='for ignored object, apply mask on the '
                                  'rectangular region or just center point.')
    self.parser.add_argument('--kitti_split', default='3dop',
                             help='different validation split for kitti: '
                                  '3dop | subcnn')

    # loss
    self.parser.add_argument('--mse_loss', action='store_true',
                             help='use mse loss or focal loss to train '
                                  'keypoint heatmaps.')
    # ctdet
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    # multi_pose
    self.parser.add_argument('--hp_weight', type=float, default=1,
                             help='loss weight for human pose offset.')
    self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                             help='loss weight for human keypoint heatmap.')
    # ddd
    self.parser.add_argument('--dep_weight', type=float, default=1,
                             help='loss weight for depth.')
    self.parser.add_argument('--dim_weight', type=float, default=1,
                             help='loss weight for 3d bounding box size.')
    self.parser.add_argument('--rot_weight', type=float, default=1,
                             help='loss weight for orientation.')
    self.parser.add_argument('--peak_thresh', type=float, default=0.2)
    
    # task
    # ctdet
    self.parser.add_argument('--norm_wh', action='store_true',
                             help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
    self.parser.add_argument('--dense_wh', action='store_true',
                             help='apply weighted regression near center or '
                                  'just apply regression on center point.')
    self.parser.add_argument('--cat_spec_wh', action='store_true',
                             help='category specific bounding box size.')
    self.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')
    # exdet
    self.parser.add_argument('--agnostic_ex', action='store_true',
                             help='use category agnostic extreme points.')
    self.parser.add_argument('--scores_thresh', type=float, default=0.1,
                             help='threshold for extreme point heatmap.')
    self.parser.add_argument('--aggr_weight', type=float, default=0.0,
                             help='edge aggregation weight.')
    # multi_pose
    self.parser.add_argument('--dense_hp', action='store_true',
                             help='apply weighted pose regression near center '
                                  'or just apply regression on center point.')
    self.parser.add_argument('--not_hm_hp', action='store_true',
                             help='not estimate human joint heatmap, '
                                  'directly use the joint offset from center.')
    self.parser.add_argument('--not_reg_hp_offset', action='store_true',
                             help='not regress local offset for '
                                  'human joint heatmaps.')
    self.parser.add_argument('--not_reg_bbox', action='store_true',
                             help='not regression bounding box size.')
    
    # ground truth validation
    self.parser.add_argument('--eval_oracle_hm', action='store_true', 
                             help='use ground center heatmap.')
    self.parser.add_argument('--eval_oracle_wh', action='store_true', 
                             help='use ground truth bounding box size.')
    self.parser.add_argument('--eval_oracle_offset', action='store_true', 
                             help='use ground truth local heatmap offset.')
    self.parser.add_argument('--eval_oracle_kps', action='store_true', 
                             help='use ground truth human pose offset.')
    self.parser.add_argument('--eval_oracle_hmhp', action='store_true', 
                             help='use ground truth human joint heatmaps.')
    self.parser.add_argument('--eval_oracle_hp_offset', action='store_true', 
                             help='use ground truth human joint local offset.')
    self.parser.add_argument('--eval_oracle_dep', action='store_true', 
                             help='use ground truth depth.')
    
    # grasp keypoints
    self.parser.add_argument('--ori_clf_mode', action="store_true", 
                            help="Detect the keypoints based on the orientation classification. This can handle the overlaping of the grasp centers."\
                              "If False, then will use the CenterNet method to regress the kpt coordinates.")
    self.parser.add_argument('--ori_num', type=int, default=20,
                            help="The orientation class number.")
    self.parser.add_argument('--dense_kpts', action='store_true',
                             help='apply weighted pose regression near center ')
    self.parser.add_argument('--kpt_type', type=str, default="hedron", 
                            choices=["hedron", "box", "tail"],
                            help='The grasp keypoint type.')
    self.parser.add_argument('--min_open_width', type=float, default=None,
                            help="The minimum open_width. If None, will not set the minimum width." \
                              "If --open_width_canonical is not None, this option will be ignored.")
    self.parser.add_argument('--open_width_canonical', default=None, type=float, 
                            help="The canonical grasp open width."\
                              "If None, all grasps in the dataset will be set to this width.")
    self.parser.add_argument('--no_kpts_refine', action='store_true', 
                            help="not employ the kpts refinement branch (including the kpt heatmap & kpt offsets), "
                            "directly use the kpts-center offset")
    self.parser.add_argument('--correct_rl', action='store_true', 
                            help="Correct the pose and the keypoint coordinates so that the left keypoint is on the left side of the image." \
                            "for each image")

    # train
    self.parser.add_argument('--skip_duplicate_center', default=None, type=float, 
                            help="Skip the duplicated center during the dataset GT generation."
                            "This step is necessary for the Vanilla CenterNet.")
    self.parser.add_argument('--no_collide_filter', action="store_true",
                            help="Train also on the grasps that cause collision")
    self.parser.add_argument('--w_weight', type=float, default=10.,
                            help = "the width loss weight.")
    self.parser.add_argument('--hm_kpts_weight', type=float, default=1,
                             help='loss weight for grasp keypoint heatmap.')
    self.parser.add_argument('--kpts_center_weight', type=float, default=1,
                             help='loss weight for kpts-center offset.')
    self.parser.add_argument('--vpt_loss_center_weight', type=float, default=-1,
                             help='loss weight of the vanishing point loss for training the center-kpts offsets.'
                              "If negative, will omit the loss"
                             )
    self.parser.add_argument('--vpt_loss_kpts_weight', type=float, default=-1,
                             help='loss weight of the vanishing point loss for training the kpts offsets.'
                              "If negative, will omit the loss"
                             )

    # test
    self.parser.add_argument('--no_nms', action="store_true", 
                            help="No NMS for the center heatmap")
    self.parser.add_argument('--center_thresh', type=float, default=0.1,
                             help='threshold for centermap.')
    self.parser.add_argument("--kpts_hm_thresh", type=float, default=0.1,
                            help="The keypoints heatmap threshold. "
                            "Specially, If set to 1, will omit the keypoint refinement branch")
    self.parser.add_argument("--test_num_per_shape", type=float, default=-1,
                            help="Sample number for each primitive shape for the testing")
    
    # for the unit test
    self.parser.add_argument('--unitTest', action="store_true", 
                            help="The unit test on the ps_grasp")
    
    # grasp GT validation
    self.parser.add_argument('--eval_oracle_kpts_center', action='store_true', 
                             help='use ground truth grasp keypoints to center offsets.')
    self.parser.add_argument('--eval_oracle_hmkpts', action='store_true', 
                             help='use ground truth human joint heatmaps.')
    self.parser.add_argument('--eval_oracle_kpts_offset', action='store_true', 
                             help='use ground truth human joint local offset.')
    
    # grasp open width
    self.parser.add_argument('--w_loss', type=str, default='l1', choices=['l1', 'sl1', 'l2'],
                            help="The loss for the. sl1 | l1 | l2")

    # the PnP algorithms  
    self.parser.add_argument('--pnp_type', type=str, default="planar",
                            choices=["cvEPnP", "cvP3P", "cvIPPE", "planar"],
                            help="The PnP algorithm to use.")
    # grasp pose evaluation
    self.parser.add_argument('--angle_th', type=float, default=45,
                            help="The threshold to determine the alignment between two SO(3) elements in degrees")
    self.parser.add_argument('--dist_th', type=float, default=0.02,
                            help="The threshold to determine the alignment between two R^(3) elements in meters")
    self.parser.add_argument('--reproj_error_th', type=float, default=None,
                            help="The threshold for filtering the pose estimation with large reprojection error")
    self.parser.add_argument('--vis_results', action="store_true", 
                              help="Visualize the detection results. Will show (a) The scene with the object, grasps, and the camera." \
                                "(b) camera view of the scene with the grasps.")
    self.parser.add_argument('--rot_sample_num', type=int, default=-1, 
                            help="The number of rotation samples for the evaluation. If both this and the trl_sample_num are None, "
                            "will load the stored grasps from the dataset")
    self.parser.add_argument('--trl_sample_num', type=int, default=-1, 
                            help="The number of rotation samples for the evaluation. If both this and the rot_sample_num are None,"
                            "will load the stored grasps from the dataset")

    # GT evaluation
    self.parser.add_argument('--test_oracle_kpts', action="store_true",
                            help="Use the ground truth kpt 2d projections for the grasp pose evaluation")
    self.parser.add_argument('--Gaussian_noise_stddev', default=0, type=float, 
                            help="The added Gaussian Noise standard deviation. Defaults to 0, which means no noise.")
    self.parser.add_argument('--use_center', action="store_true", 
                            help="Use the center in the PnP process, which makes it a 5-points problem.")
  
    # for inspection and debug
    self.parser.add_argument('--inspect_aug', action="store_true", 
                            help="Inspect the data augmentation. Will display relevant information.")


    # baseline
    self.parser.add_argument("--baseline_method", type=str, default="GPG", choices=["GPG", "GraspNet"],
                            help="The baseline methods to test")
    self.parser.add_argument("--load_exist_baseResults", action="store_true", 
                          help="whether load the pre-saved results")

    # baseline - GPG
    self.parser.add_argument("--show_final_grasp", action="store_true")
    self.parser.add_argument("--tray_grasp", action="store_true")
    self.parser.add_argument("--using_mp", action="store_true")

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    # added for the grasp pose
    opt.kpts_refine = not opt.no_kpts_refine
    if opt.vpt_loss_center_weight > 0  or opt.vpt_loss_kpts_weight > 0:
      assert opt.kpt_type == "box", \
        "Only the box type keypoint support the vanishing loss"
      opt.vpt_loss = True
    else:
      opt.vpt_loss = False

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')

  
    #### For the grasp pose. #######
    # Skip overlapping center for the vanilla CEnterNet. 
    opt.skip_duplicate_center = True if not opt.ori_clf_mode \
      else False

    # the input modality - This will influence the normalization in the detector's preprocess function ( for test data )
    opt.input_mod = "RGD" if opt.task == "grasp_pose" else "BGR"

    # the test angular threshold - degree to radians
    opt.angle_th = opt.angle_th * np.pi / 180.

    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    
    if opt.task == 'exdet':
      # assert opt.dataset in ['coco']
      num_hm = 1 if opt.agnostic_ex else opt.num_classes
      opt.heads = {'hm_t': num_hm, 'hm_l': num_hm, 
                   'hm_b': num_hm, 'hm_r': num_hm,
                   'hm_c': opt.num_classes}
      if opt.reg_offset:
        opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
    elif opt.task == 'ddd':
      # assert opt.dataset in ['gta', 'kitti', 'viper']
      opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
      if opt.reg_bbox:
        opt.heads.update(
          {'wh': 2})
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'ctdet':
      # assert opt.dataset in ['pascal', 'coco']
      opt.heads = {'hm': opt.num_classes,
                   'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'multi_pose':
      # assert opt.dataset in ['coco_hp']
      opt.flip_idx = dataset.flip_idx
      opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
      if opt.hm_hp:
        opt.heads.update({'hm_hp': 17})
      if opt.reg_hp_offset:
        opt.heads.update({'hp_offset': 2})
    elif opt.task == "grasp_pose":
      if opt.ori_clf_mode:
        opt.heads = {
          'hm': opt.ori_num,                                            # the number of orientation classes
          'w': opt.ori_num,                      # 1 width classe for each ori_cls
          'kpts_center_offset': 2 * dataset.num_grasp_kpts * opt.ori_num,   # (2(L) + 2(R) + 2(TL) + 2(TR))*4
          "reg": 2,
        }
      else:
        opt.heads = {
          'hm': 1, 
          "w": 1, 
          "kpts_center_offset": 2 * dataset.num_grasp_kpts,
          "reg": 2,
        }
      # Below is for the kpt-refinement branch
      if opt.kpts_refine:
        opt.heads.update({"hm_kpts": 4})
        opt.heads.update({"kpts_offset": 2})
    elif opt.task == "grasp_baseline_GPG" or opt.task == "grasp_baseline_GraspNet":
      opt.heads = {}
      pass
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
      'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'exdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'multi_pose': {
        'default_resolution': [512, 512], 'num_classes': 1, 
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                     [11, 12], [13, 14], [15, 16]]},
      'ddd': {'default_resolution': [384, 1280], 'num_classes': 3, 
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'kitti'},
      # TODO: below is just a copy of the multi_pose.
      'grasp_pose': {
        'default_resolution': [512, 512], 'num_classes': 1,
        'mean': [0.850092, 0.805317, 0.247344], 'std': [0.104114, 0.113242, 0.089067],
        "dataset": "ps_grasp", 'num_grasp_kpts': 4,
      }
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt

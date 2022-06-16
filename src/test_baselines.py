from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyparsing import col

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation as R

#from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.keypoints import kpts_3d_to_2d, plot_grasps_kpts
from utils.utils import AverageMeter
from utils.vis import construct_scene_with_grasp_preds
from datasets.dataset_factory import dataset_factory

try:
    import dex_net.apps.gpg as gpg
except:
    gpg = None
    print("The environment for testing the GPG baseline is not ready.")

try:
    from utils.utils_graspnet import addGraspNetOpts, frame_trf_graspnet2ps
    import graspnet_6dof.utils.utils as graspnet_utils
    import graspnet_6dof.grasp_estimator as grasp_estimator
except:
    addGraspNetOpts = lambda opts: opts
    print("The environment for testing the GraspNet baseline is not ready.")

def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    if opt.baseline_method == "GPG":
        opt = gpg.add_GPG_config(opt)
    print(opt)
    Logger(opt)

    # dataset
    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    num_iters = len(dataset)
    
    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_timer = AverageMeter()

    # shape counter
    shape_counter = {
        'cuboid': 0,
        'cylinder': 0,
        'stick': 0,
        'ring': 0,
        'sphere': 0,
        'semi_sphere': 0
    }

    # path 
    file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(file_path))
    base_results_dir = os.path.join(root_dir, "base_results_{}".format(opt.baseline_method))
    if not os.path.exists(base_results_dir):
        os.mkdir(base_results_dir)
    
    # prepare grasp detector
    if opt.baseline_method == "GraspNet":
        grasp_sampler_args = graspnet_utils.read_checkpoint_args(args.grasp_sampler_folder)
        grasp_sampler_args.is_train = False
        # grasp_sampler_args.which_epoch = 1 # train longer seems overfitting
        grasp_evaluator_args = None # no evaluation or refine
        estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                       grasp_evaluator_args, args)

    for img_id in range(num_iters):

        np.random.seed(img_id)      # for different object color

        # debug: - For the scene 145
        #if not ind in [196, 197, 198, 199]:
        #     contin

        # get the scene and camera index
        scene_idx = dataset.scene_idxs[img_id]
        cam_idx = dataset.camera_idxs[img_id]

        # get the camera intrinsic and the oracle open width
        intrinsic, camera_poses, obj_types, obj_dims, obj_poses, grasp_poses, grasp_widths, grasp_collisions,  = \
            dataset._get_scene_info(scene_idx)


        # # for debug:
        # if not "cuboid" in obj_types:
        #     continue
        
        # limit the testing number for debugging
        if opt.test_num_per_shape > 0:
            if all(shape_counter[obj_type] >= opt.test_num_per_shape for obj_type in ["cuboid", "cylinder", "stick", "ring", "sphere", "semi_sphere"]):
                break
            elif shape_counter[obj_types[0]] >= opt.test_num_per_shape:
                continue
            else:
                shape_counter[obj_types[0]] += 1
        
        # determine the save path
        save_path = os.path.join(base_results_dir, "{}.json".format(img_id))
            
        #### Obtain the baseline results
        results_this = {}
        locations = []
        quaternions = []
        widths = []

        if opt.baseline_method == "GPG":
            grasp_pred_frame = "tabletop"
        elif opt.baseline_method == "GraspNet":
            grasp_pred_frame = "camera"
        else:
            raise NotImplementedError

        if not opt.load_exist_baseResults:
            _, _, widths_gt = \
                dataset._get_gt_grasps(scene_idx, cam_idx, filter_collides=True, center_proj=False, correct_rl=opt.correct_rl)
            widths_avg = np.mean(np.concatenate(widths_gt))
            if opt.baseline_method == "GPG":
                opt.gripper.hand_outer_diameter = widths_avg + 2 * opt.gripper.finger_width

                # get the object point cloud in the tabletop frame
                pc = dataset.get_pc(img_id, frame="tabletop", flatten=True)
                pc_obj = pc[pc[:, 2] > 0.005]

                # get the grasps
                cam_table_trl = camera_poses[cam_idx, :, :]
                cam_table_trl = cam_table_trl[:3, 3]
                start_time = time.time()
                grasps, _ = gpg.get_grasps(pc_obj, cam_table_trl, opt)
                end_time = time.time()
                time_this = end_time - start_time 
                quaternions, locations = gpg.grasp_to_pose(grasps, opt)
                widths = [opt.gripper.hand_outer_diameter-2*opt.gripper.finger_width] * quaternions.shape[0]

            elif opt.baseline_method == "GraspNet":
                # get the object point cloud in the camera frame
                pc = dataset.get_pc(img_id, frame="camera", flatten=True)
                pc_table = dataset.get_pc(img_id, frame="tabletop", flatten=True)
                pc = pc[pc_table[:, 2] > 0.005]


                # generate grasps - format List[array(4, 4)]
                start_time = time.time()
                grasps, _ = estimator.generate_and_refine_grasps(pc)
                end_time = time.time()
                grasps = np.stack(grasps, axis=0)

                time_this = end_time - start_time


                # parse the result
                grasps = frame_trf_graspnet2ps(grasps)
                rot_mats = grasps[:, :3, :3]
                rs = R.from_matrix(rot_mats)
                quaternions = rs.as_quat()
                locations = grasps[:, :3, 3]

                # just use the average widths
                widths = [widths_avg] * quaternions.shape[0]


            results_this["locations"] = np.array(locations)
            results_this["quaternions"] = np.array(quaternions)
            results_this['widths'] = np.array(widths) #* 3
            results_this['time'] = time_this
            save_results(results_this, save_path, verbose=True)

        else:
            # directly load the results
            results_this = load_results(save_path)
            time_this = results_this['time']

        # store in the results
        results[img_id] = results_this

        # update the timer
        avg_timer.update(val=time_this, n=1)

        # visualize results
        if opt.vis_results or opt.debug == 5:
            
            # evaluate this result
            results_tmp = {}
            results_tmp[img_id] = results_this
            pred_succ = dataset.run_eval(results_tmp, opt.save_dir, angle_th=opt.angle_th, dist_th=opt.dist_th, \
                rot_sample=opt.rot_sample_num, trl_sample=opt.trl_sample_num, pred_frame=grasp_pred_frame)[img_id]

            s = construct_scene_with_grasp_preds(
                obj_types=obj_types,
                obj_dims=obj_dims,
                obj_poses=obj_poses,
                # obj_colors= [np.random.choice(range(256), size=3)],
                camera_pose=camera_poses[cam_idx],
                grasp_results=results_this,
                grasp_color=[0, 0, 255],
                grasp_succ=pred_succ,
                max_grasp_num=40,
                grasp_pred_frame=grasp_pred_frame
            )
            # s.vis_scene()

            # load image
            img_path = dataset.images[img_id]
            color = cv2.imread(img_path)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            # # see the GT
            # poses_gt, _, widths_gt = \
            #     dataset._get_gt_grasps(scene_idx, cam_idx, filter_collides=True, center_proj=False, correct_rl=opt.correct_rl)
            # results_gt = {
            #     "poses": np.concatenate(poses_gt, axis=0), 
            #     "widths": np.concatenate(widths_gt, axis=0)
            # }
            # s_gt = construct_scene_with_grasp_preds(
            #     obj_types=obj_types,
            #     obj_dims=obj_dims,
            #     obj_poses=obj_poses,
            #     camera_pose=camera_poses[cam_idx],
            #     grasp_results=results_gt,
            #     grasp_color=[0, 0, 255],
            # )
            # s_gt.vis_scene(mode = "trimesh")


            if opt.debug == 5:
                # save the 3d grasp
                colors, _ = s.render_imgs(instance_masks=False, grasp_mode=0)
                cv2.imwrite(
                    os.path.join(opt.debug_dir, '{}_grasp_pred.png'.format(img_id)),
                    colors[0][:,:,::-1]
                )

            if opt.vis_results:

                # visualize the scene with the grasps
                s.vis_scene("trimesh") 
        bar.next()
    bar.finish()

    print("\n The average time taken for the {} method: {:4f} SPF ({:4f} FPS). ".format(opt.baseline_method, avg_timer.avg, 1./avg_timer.avg))
    dataset.run_eval(results, opt.save_dir, angle_th=opt.angle_th, dist_th=opt.dist_th, \
        rot_sample=opt.rot_sample_num, trl_sample=opt.trl_sample_num, pred_frame=grasp_pred_frame)
    


def load_results(path):
    with open(path) as d:
        result = json.load(d)
    
    for key in result.keys():
        if isinstance(result[key], list):
            result[key] = np.array(result[key])
    return result

def save_results(result, path, verbose=False):
    if verbose:
        print("Saving the current result to: {}".format(path))
    result_tmp = copy.deepcopy(result)
    for key in result_tmp.keys():
        if isinstance(result_tmp[key], np.ndarray):
            result_tmp[key] = result_tmp[key].tolist()
    with open(path, 'w') as fp:
        json.dump(result_tmp, fp)

if __name__ == '__main__':
    opt = opts()
    opt = addGraspNetOpts(opt)
    args = opt.parse()
    if hasattr(args, "sample_batch_size"):
        args.batch_size = args.sample_batch_size    # that is what the grasp estimator use

    test(args)


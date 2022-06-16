from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R
import trimesh

# NOTE: mayavi environment requires opencv and the opencv-contrib cannot work.
# due to its pickiness it is better to create a separate environment for it and concentrate the related code.
from mayavi import mlab



# set the path
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import _init_paths

from data_generation.grasp.grasp import Grasp
from utils.ddd_utils import depth2pc
from physical.utils_physical import GraspPoseRefineScale


# globle setting
root_path = os.path.join(os.path.dirname(__file__), "3d_grasp_phy")

def draw_scene(pc,
               grasps=[],
               widths = None,
               grasp_scores=None,
               grasp_color=None,
               gripper_color=(0, 1, 0),
               grasps_selection=None,
               visualize_diverse_grasps=False,
               pc_color=None,
               plasma_coloring=False):
    """
    Draws the 3D scene for the object and the scene.
    Args:
      pc: point cloud of the object
      grasps: list of 4x4 numpy array indicating the transformation of the grasps.
        grasp_scores: grasps will be colored based on the scores. If left 
        empty, grasps are visualized in green.
      grasp_color: if it is a tuple, sets the color for all the grasps. If list
        is provided it is the list of tuple(r,g,b) for each grasp.
      mesh: If not None, shows the mesh of the object. Type should be trimesh 
         mesh.
      show_gripper_mesh: If True, shows the gripper mesh for each grasp. 
      grasp_selection: if provided, filters the grasps based on the value of 
        each selection. 1 means select ith grasp. 0 means exclude the grasp.
      visualize_diverse_grasps: sorts the grasps based on score. Selects the 
        top score grasp to visualize and then choose grasps that are not within
        min_seperation_distance distance of any of the previously selected
        grasps. Only set it to True to declutter the grasps for better
        visualization.
      pc_color: if provided, should be a n x 3 numpy array for color of each 
        point in the point cloud pc. Each number should be between 0 and 1.
      plasma_coloring: If True, sets the plasma colormap for visualizting the 
        pc.
    """

    max_grasps = 100
    grasps = np.array(grasps)

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if len(grasps) > max_grasps:

        print('Downsampling grasps, there are too many')
        chosen_ones = np.random.randint(low=0,
                                        high=len(grasps),
                                        size=max_grasps)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]

    if pc_color is None and pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc[:, 2],
                          colormap='plasma')
        else:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          color=(0.1, 0.1, 1),
                          scale_factor=0.01)
    elif pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc_color[:, 0],
                          colormap='plasma')
        else:
            rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            rgba[:, :3] = np.asarray(pc_color)
            rgba[:, 3] = 255
            src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            src.add_attribute(rgba, 'colors')
            src.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(src)
            g.glyph.scale_mode = "data_scaling_off"
            g.glyph.glyph.scale_factor = 0.01

    if grasp_scores is not None:
        indexes = np.argsort(-np.asarray(grasp_scores))
    else:
        indexes = range(len(grasps))

    print('draw scene ', len(grasps))

    selected_grasps_so_far = []
    removed = 0

    if grasp_scores is not None:
        min_score = np.min(grasp_scores)
        max_score = np.max(grasp_scores)
        top5 = np.array(grasp_scores).argsort()[-5:][::-1]

    for ii in range(len(grasps)):
        i = indexes[ii]
        if grasps_selection is not None:
            if grasps_selection[i] == False:
                continue
        if widths is not None:
            w = widths[ii]
        else:
            w = 0.1

        g = grasps[i]
        gripper = Grasp(open_width=w, pose=g)
        gripper_meshes = gripper.get_mesh()
        gripper_mesh = trimesh.util.concatenate(gripper_meshes)

        gripper_color = (0.0, 0.0, 1.0)
        # gripper_mesh = sample.Object(
            # 'gripper_models/panda_gripper.obj').mesh
        # gripper_mesh.apply_transform(g)
        mlab.triangular_mesh(
            gripper_mesh.vertices[:, 0],
            gripper_mesh.vertices[:, 1],
            gripper_mesh.vertices[:, 2],
            gripper_mesh.faces,
            color=gripper_color,
            opacity=1 if visualize_diverse_grasps else 0.5)

    print('removed {} similar grasps'.format(removed))



def load_results(rgb_path, dep_path, poses_path):
    rgb = cv2.imread(rgb_path)
    dep = np.load(dep_path)
    with np.load(poses_path) as poses:
        intrinsic = poses["intrinsic"]
        grasp_poses_pred_cam = poses["grasp_pose_pred_cam"]
        widths = poses["widths"]
    
    return rgb, dep, intrinsic, grasp_poses_pred_cam, widths

def get_paths(dir_name):
    ls = os.listdir(dir_name)
    rgb_paths = []
    dep_paths = []
    poses_paths = []

    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in ["png"]:
            # get the id
            file_name_noExt = file_name[:file_name.rfind('.')]
            id = file_name_noExt[file_name_noExt.rfind('_')+1:]

            # get the other paths
            dep_name = "depth_raw_"+id+".npy"
            poses_name = "poses_"+id+".npz"

            # add to cache
            rgb_paths.append(os.path.join(dir_name, file_name))
            dep_paths.append(os.path.join(dir_name, dep_name))
            poses_paths.append(os.path.join(dir_name, poses_name))
        
    return rgb_paths, dep_paths, poses_paths
 


def main(args):
    # The directory and get the paths
    result_dir = os.path.join(root_path, args.exp_name)
    rgb_paths, dep_paths, poses_paths = get_paths(result_dir)
    
    if args.refine_poses:
        pose_refiner = GraspPoseRefineScale(intrinsic=None)

    # check results one by one
    for rgb_path ,dep_path, poses_path in zip(rgb_paths, dep_paths, poses_paths):
        rgb, dep, intrinsic, grasp_poses_pred_cam, widths = load_results(rgb_path,dep_path, poses_path)

        # refine poses
        if args.refine_poses:
            pose_refiner.store_info(dep=dep, intrinsic=intrinsic)
            grasp_poses_pred_cam, refine_succ = pose_refiner.refine_poses(grasp_poses_pred_cam)
            grasp_poses_pred_cam = grasp_poses_pred_cam[refine_succ, :, :]
        
        # get the point cloud
        pcl_cam = depth2pc(dep, intrinsic, frame="camera", flatten=True)
        print(grasp_poses_pred_cam.shape)
        draw_scene(pcl_cam, grasps=grasp_poses_pred_cam, pc_color=rgb.reshape(-1, 3))
        print("Close the window to see the next")
        mlab.show()

        # visualize

    return


def get_args():
    opt = argparse.ArgumentParser()
    opt.add_argument("--exp_name", type=str, default="test")
    opt.add_argument("--refine_poses", action="store_true")

    args = opt.parse_args()
    return args


if __name__=="__main__":
    args = get_args()
    main(args)

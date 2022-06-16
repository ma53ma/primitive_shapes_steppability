import os
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt2d


# set the import path
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import _init_paths
from data_generation.grasp.grasp import Grasp
from utils.transform import create_homog_matrix
from utils.keypoints import plot_grasps_kpts


def quad2homog(locations, quaternions):
    locations = locations.reshape((-1, 3))
    quaternions = quaternions.reshape((-1, 4))
    N_grasps = locations.shape[0]
    poses = np.zeros((N_grasps, 4, 4), dtype=float)
    for i in range(N_grasps):
        r = R.from_quat(quaternions[i, :])
        pose = create_homog_matrix(R_mat=r.as_matrix(), T_vec=locations[i, :])
        poses[i, :, :] = pose 
    return poses


def draw_kpts(rgb, kpts_2d_pred, opt, sample_num = -1):
    N_grasps = kpts_2d_pred.shape[0]
    if N_grasps == 0:
        return rgb 
    if sample_num > 0:
        sample_num = min(sample_num, N_grasps)
        ids = np.random.choice(N_grasps, sample_num, replace=False)
        kpts_draw = kpts_2d_pred[ids, :, :]
    else:
        kpts_draw = kpts_2d_pred
    img_kpts_pred = plot_grasps_kpts(rgb, kpts_draw, kpts_mode=opt.kpt_type, size=5)

    return img_kpts_pred


class GraspPoseRefineBase():
    def __init__(self) -> None:
        pass

    def store_info(self):
        """Different refinement approach might requires different materials (e.g. depth map)
        Store them here
        """
        raise NotImplementedError()

    def refine_poses(grasp_poses):
        raise NotImplementedError


class GraspPoseRefineScale(GraspPoseRefineBase):
    """
    Refine the grasp by scale the translation along the camera

    Args:
        intrinsic (3, 3):       The camera intrinsic matrix
    """
    def __init__(self, intrinsic) -> None:
        super().__init__()
        self.intrinsic = intrinsic

        # cache for the required materials
        self.dep = None
    
    def store_info(self, dep, intrinsic=None):
        self.dep = dep
        if intrinsic is not None:
            self.intrinsic = intrinsic

    def refine_poses(self, grasp_poses):
        """
        Refine the poses by scale the translation s.t. 
        the center between the tips aligns with the observed object align that direction 
        Args:
            grasp_poses (N, 4, 4):          The frame transformation from the camera to gripper. 
                                            Assume the gripper frame origin is the center between the tips
        """
        N = grasp_poses.shape[0]
        # preprocess
        dep = self.preprocess_dep(self.dep)

        refine_successed = np.zeros((N,), dtype=bool)
        grasp_poses_refined = np.zeros_like(grasp_poses)
        for i in range(N):
            pose = grasp_poses[i, :, :]
            trl = pose[:3, 3].reshape(-1)
            rot_mat = pose[:3, :3]

            # get the image coordinate
            img_coord = (self.intrinsic @ trl.reshape((-1, 1))).reshape(-1)
            img_coord = (img_coord / img_coord[-1])[:2]

            # get the depth - OpenCV coordinate
            depth_value = dep[int(img_coord[1]), int(img_coord[0])]

            # scale 
            if(np.linalg.norm(trl)==0):
                print("The pose has problem!")
            elif (depth_value == 0):
                print("Depth has problem")
            else:
                refine_successed[i] = True
            trl_refined = trl * depth_value / np.linalg.norm(trl)

            # assemble to get the new pose
            grasp_poses_refined[i, :3, 3] = trl_refined
            grasp_poses_refined[i, :3, :3] = rot_mat
            grasp_poses_refined[i, 3, 3] = 1.

        return grasp_poses_refined, refine_successed

    
    def preprocess_dep(self, dep):
        # median filter
        scale = 0.01
        dep_scaled = (dep / scale).astype(np.int32)
        dep_scaled = medfilt2d(dep_scaled, kernel_size=5)
        dep = dep_scaled.astype(np.float32) * scale

        return dep
      

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys

from torch import select

# for using ros in the python3
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")
import rospy
import rospkg
from std_msgs.msg import Float64MultiArray
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# try:
#     from mayavi import mlab
# except:
#     mlab = None


# set the path
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
import _init_paths


from detectors.detector_factory import detector_factory
from pose_recover.pnp_solver_factory import PnPSolverFactor
from opts import opts
from utils.keypoints import kpts_3d_to_2d, plot_grasps_kpts
from utils.transform import create_homog_matrix, create_rot_mat_axisAlign
from utils.ddd_utils import depth2pc
from data_generation.grasp.grasp import Grasp

from physical.utils_physical import quad2homog, draw_kpts, GraspPoseRefineScale

# camera package
import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.display import display_images_cv, display_rgb_dep_cv, wait_for_confirm


# options
calib_M_CL = False 
ROOT_DIR = "/home/cyy/3d_grasp_phy"
EXP_NAME = "test"


##########################################################################
##      Global Variable
##########################################################################
# grasp configuration publisher
pub_grasp = None

# frame transformation from the camera to the aruco tag (default)
M_CL = np.array([[-0.57073818, -0.80965115, 0.13683183, -0.11441497],
                [-0.64278335,  0.33683378, -0.68802077,  0.15725438],
                [ 0.51096722, -0.48063294, -0.71267417,  0.64775689],
                [ 0.,          0.,          0. ,         1.        ]] )

# frame transformation from the robot base to aruco tag
M_BL = np.array([[1., 0., 0.,  0.30000],
                 [0., 1., 0.,  0.32000],
                 [0., 0., 1.,  -0.09],
                 [0., 0., 0.,  1.00000]])


def parse_results(opt, ret, pnp_solver):
    """
    Parse the keypoints detection network results.
    Apply the PnP algorithm to obtain the grasp poses in the camera frame represented as the translations and quaternion rotations.

    Returns:
        kpts_2d_pred (N, 4, 2).     The keypoint coordinates
        locations (N, 3).           The translation 
    """
    # parse the results
    kpts_2d_pred = np.empty((0, 4, 2))
    locations = np.empty((0, 3))
    quaternions = np.empty((0, 4))
    widths = np.empty((0,))

    dets = np.array(ret["results"][1])
    if dets.size != 0:
        kpts_2d_pred_init = dets[:, 2:10].reshape(-1, 4, 2)
        centers_2d_pred_init = dets[:, :2].reshape(-1, 1, 2)
        widths_pred = dets[:, 10]

        # filter by scores
        scores = dets[:, 11]
        kpts_2d_pred_init = kpts_2d_pred_init[scores > opt.center_thresh]
        widths_pred = widths_pred[scores > opt.center_thresh]

        # recover the pose results
        locations = []
        quaternions = []
        widths = []
        kpts_2d_pred = []
        centers_2d_pred = []
        reproj_errors = []
        center_scores = []

        N_grasps = kpts_2d_pred_init.shape[0]
        for i in range(N_grasps):

            if opt.open_width_canonical is not None:
                proj_width = opt.open_width_canonical
            elif opt.min_open_width is not None:
                proj_width = widths_pred[i] if widths_pred[i] > opt.min_open_width \
                    else opt.min_open_width
            else:
                proj_width =  widths_pred[i]    
            pnp_solver.set_open_width(open_width=proj_width)

            try:
                location, quaternion, projected_points, reprojectionError = \
                    pnp_solver.solve_pnp(
                        kpts_2d_pred_init[i, :, :],
                        centers_2d_pred_init[i, :, :]
                    )
            except:
                location = None
                quaternion = None

            # skip if the grasp pose recovery failed
            if location is None or quaternion is None:
                continue

            if opt.reproj_error_th is None or reprojectionError < opt.reproj_error_th:
                locations.append(location)
                quaternions.append(quaternion)
                widths.append(widths_pred[i])
                kpts_2d_pred.append(kpts_2d_pred_init[i, :, :])
                centers_2d_pred.append(centers_2d_pred_init[i, :, :])
                reproj_errors.append(reprojectionError)
                center_scores.append(scores[i])

        # return the results
        locations = np.array(locations)
        quaternions = np.array(quaternions)
        widths = np.array(widths) #* 3
        kpts_2d_pred = np.array(kpts_2d_pred)
        centers_2d_pred = np.array(centers_2d_pred)
        reproj_errors = np.array(reproj_errors)
        center_scores = np.array(center_scores)


    return kpts_2d_pred, centers_2d_pred, widths, locations, quaternions, reproj_errors, center_scores

def sort_grasp(results):
    """
    for now only pick one
    """
    kpts_2d_pred = results["kpts_2d"]
    centers_2d_pred = results["centers_2d"]
    widths = results["widths"]
    locations = results["locations"]
    quaterions = results["quaternions"]
    reproj_errors = results["reproj_errors"]        # (N, )

    #  score with the least reprojection error
    score_1 = - reproj_errors
    beta_1 = 10.
    score_2 = locations
    beta_2 = 1.

    score = score_1

    # sort the score in descend order
    idx_sort = np.array(score).argsort()[::-1]
    loc_sort = locations[idx_sort, :]
    quat_sort = quaterions[idx_sort, :]
    w_sort = widths[idx_sort]

    return idx_sort, loc_sort, quat_sort, w_sort

def grasp_pose_2_msg(grasp_pose, width):
    """
    grasp_pose: (4, 4)
    Return:
        msg (Float64MultiArray):        Format is (7, ), where first three is translation, final 4 is quaternion in (x, y, z, w).  
                                        This is for the MoveIt
    """
    msg = Float64MultiArray()
    # msg.data = grasp_pose.reshape(-1)
    # TODO: convert to whatever is taken by the move it. Move group accept quaternion
    trl = grasp_pose[:3, 3]
    rot_mat = grasp_pose[:3, :3]
    r = R.from_matrix(rot_mat)
    quat = r.as_quat()

    msg.data = np.concatenate((trl, quat, np.array(width).reshape(1, )), axis=0)
    return msg

def pose_cam_2_robot(pose_cam):
    """
    Args:
        pose_cam: (4, 4).           The transform from the camera frame to a target frame.
    Returns:
        pose_robot: (4, 4). i.e.    The frame transform from robot to grasp
    """
    global M_CL, M_BL
    pose_robot = M_BL @ np.linalg.inv(M_CL) @ pose_cam
    return pose_robot


def frame_trf_ps2handy(poses):
    """convert the pose from the network coordinate system to the real(Handy robot) system"""
    rot_mat = create_rot_mat_axisAlign(align_order=[3, -2, 1])
    trf = create_homog_matrix(R_mat=rot_mat)
    return poses@trf


class PhyResultSaver():
    def __init__(self, root_dir, exp_name) -> None:
        self.root_dir = root_dir
        self.exp_name = exp_name
        self.save_dir = os.path.join(self.root_dir, self.exp_name)
        self.idx = 0


        # build directory
        if not os.path.exists(self.root_dir):
            print("Creating the root dir for save out: {}".format(self.root_dir))
            os.mkdir(self.root_dir)
        if not os.path.exists(self.save_dir):
            print("Creating the save out dir for this exp: {}".format(self.save_dir))
            os.mkdir(self.save_dir)
        
        # get the start index
        self.get_start_id()
            
        pass

    def get_start_id(self):
        # load all png file
        image_ext = ["png"]
        ls = os.listdir(self.save_dir)
        color_imgs = []
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                color_imgs.append(os.path.join(self.save_dir, file_name))
        
        # initialize the start idx
        self.idx = len(color_imgs)

        print("There are {} exist results. Start from idx {}".format(self.idx, self.idx))


    def save_results(self, rgb, dep, intrinsic, grasp_pose_cam, widths):
        """
        Save out the results
        Args:
            rgb (H, W, 3):              rgb image. Saved as color_img_[idx].png
            depth (H, W):               raw depth map. Saved as depth_raw_[idx].npy
            intrinsic (4, 4):           camera intrinsic matrix. Will be saved in the info_[idx].npy
            grasp_pose_cam (N, 4, 4):   the predicted grasp poses in the camera frame
        """
        print("Saving out the results for the index: {}".format(self.idx))
        # the paths
        color_img_path = os.path.join(self.save_dir, "color_img_{}.png".format(self.idx))
        depth_path = os.path.join(self.save_dir, "depth_raw_{}.npy".format(self.idx))
        poses_path = os.path.join(self.save_dir, "poses_{}.npz".format(self.idx))

        # save
        np.save(depth_path, dep)
        cv2.imwrite(color_img_path, rgb[:,:,::-1])
        np.savez(
            poses_path,
            intrinsic = intrinsic,
            grasp_pose_pred_cam = grasp_pose_cam,
            widths=widths
        )

        # increment the idx
        self.idx += 1

    


def run(opt):

    global ROOT_DIR, EXP_NAME
    global M_CL, calib_M_CL


    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    PNPSolver = PnPSolverFactor[opt.pnp_type] 
    pnp_solver = PNPSolver(
        kpt_type=opt.kpt_type,
        use_center=opt.use_center
    )

    # The D435 runner & the aruco calibrator
    exposure = 200 if calib_M_CL else 520
    d435_configs = d435.D435_Configs(
        W_dep=640,
        H_dep=480,
        W_color=640,
        H_color=480,
        exposure=exposure,
        gain=35
    )

    d435_starter = d435.D435_Runner(d435_configs)

    calibrator_CtoW = CtoW_Calibrator_aruco(
        d435_starter.intrinsic_mat,
        distCoeffs=np.array([0.08847, -0.04283, 0.00134, -0.00102, 0.0]),
        markerLength_CL = 0.093,
        maxFrames = 30,
        flag_vis_extrinsic = True,
        flag_print_MCL = True,
        stabilize_version =True 
    )
    pnp_solver.set_camera_intrinsic_matrix(d435_starter.intrinsic_mat)

    # pose refiner
    pose_refiner = GraspPoseRefineScale(intrinsic=None)

    # saver
    results_saver = PhyResultSaver(root_dir=ROOT_DIR, exp_name=EXP_NAME)

    # calibrate
    if calib_M_CL:
        print("Calibrating the extrinsic matrix...")
        while not calibrator_CtoW.stable_status:
            # get frames
            rgb, dep, success = d435_starter.get_frames()
            # calibrate
            M_CL, corners_aruco, img_with_ext, status = calibrator_CtoW.process(rgb, dep) 
            display_images_cv([img_with_ext], window_name="Aruco")
            cv2.waitKey(1)
            # assert status, "The aruco tag can not be detected"        
            print(M_CL)

        print("Extrinsic matrix calibration done.")
        exit()
    
    # run
    while(True):
        rgb, dep, success = d435_starter.get_frames()
        image = rgb.copy().astype(np.float32)
        image[:, :, 2] = dep 

        # display_rgb_dep_cv(rgb, dep)
        # cv2.waitKey(1)
        # continue

        # detect
        ret = detector.run(image)

        # parse the results, and transform to the SE(3) form
        kpts_2d_pred, centers_2d_pred, widths, locations, quaternions, reproj_errors, center_scores = parse_results(opt, ret, pnp_solver)
        results = {
            "kpts_2d": kpts_2d_pred,
            "centers_2d": centers_2d_pred,
            "widths": widths,
            "locations": locations,
            "quaternions": quaternions,
            "reproj_errors": reproj_errors,
            "center_scores": center_scores
        }

        # parse the results to the se3
        grasp_poses_cam = quad2homog(
            locations=results["locations"],
            quaternions=results["quaternions"]
        )

        # plot the predicted keypoints
        img_kpts_pred = draw_kpts(rgb, kpts_2d_pred, opt, sample_num=10)
        display_rgb_dep_cv(img_kpts_pred, dep, ratio=2)
        opKey = cv2.waitKey(1)


        # verify the poses
        # pcl_cam = depth2pc(dep, d435_starter.intrinsic_mat, frame="camera", flatten=True)
        # draw_scene(pcl_cam, grasps=grasp_poses_cam, pc_color=rgb.reshape(-1, 3))
        # mlab.show()

        if opKey == ord("q"):
            break
        elif opKey ==  ord("c"):
            # save out
            results_saver.save_results(rgb, dep, d435_starter.intrinsic_mat, grasp_poses_cam, widths=widths)

            # sort 
            idx_sort, loc_sort, quat_sort, w_sort = sort_grasp(results)

            # select
            selected=False
            k = 0
            while(not selected):
                select_idx = idx_sort[k]

                # convert to the homog
                pose_pick = quad2homog(
                    locations=loc_sort[k, :], 
                    quaternions=quat_sort[k, :]
                ).squeeze()

                # transform to the Handy endeffector coordinate
                pose_pick = frame_trf_ps2handy(pose_pick)

                # refine
                pose_refiner.store_info(dep=dep, intrinsic=d435_starter.intrinsic_mat)
                pose_pick, refine_success = pose_refiner.refine_poses(pose_pick[None,:,:])
                pose_pick = pose_pick.squeeze()
                refine_success = refine_success.squeeze()
                selected = refine_success

                # convert to robot frame pose
                grasp_pose_robot = pose_cam_2_robot(pose_pick)

                # width
                w = w_sort[k]

                # increment the index
                k += 1

            # verify the selection
            img_selected_kpts = draw_kpts(rgb, kpts_2d_pred[select_idx:select_idx+1, :, :], opt, sample_num=10)
            display_rgb_dep_cv(img_selected_kpts, dep, ratio=2, window_name="The selected grasp")
            opKey = cv2.waitKey()
            cv2.destroyWindow('The selected grasp')


            # publish to the topic
            grasp_msg = grasp_pose_2_msg(grasp_pose_robot, w)
            pub_grasp.publish(grasp_msg)
    


if __name__ == '__main__':
    # test rospy
    rospy.init_node("Grasping_Experiment")
    pub_grasp = rospy.Publisher('/grasp_config', Float64MultiArray, queue_size=10)

    opt = opts().init()
    run(opt)

"""
Get the mean and the std deviation
"""
import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import _init_paths
from utils.file import read_numList_from_file, write_numList_to_file
from utils.utils import AverageMeter


dataset_root = os.path.join(os.path.dirname(__file__), "../data")

def get_args():
    parser = argparse.ArgumentParser("Get the training mean and the standard deviation statistics of a grasping dataset."
    "The dataset is assumed to be located at ../data/DATASET_FOLDER")
    parser.add_argument("dataset_folder", type=str, 
                    help="Dataset folder name. (e.g. ps_grasp_single_10k)")
    args = parser.parse_args()
    args.dataset_dir = os.path.join(dataset_root, args.dataset_folder)
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    print("\nThe dataset folder is: {}".format(args.dataset_dir))
    return args

def get_mean_std_files(color_files, depth_files):
    rgb_mean_stats = AverageMeter()
    rgb_square_stats = AverageMeter()
    depth_mean_stats = AverageMeter()
    depth_square_stats = AverageMeter()

    # get the means
    print("Calculating...")
    for rgb_f, dep_f in tqdm(zip(color_files, depth_files), total=len(color_files)):
        rgb = cv2.imread(rgb_f)[:, :, ::-1]
        rgb = rgb/255.
        dep = np.load(dep_f)
        rgb_mean_stats.update(
            np.mean(rgb.reshape(-1, 3), axis=0),
            n=1
        )
        rgb_square_stats.update(
            np.mean( (rgb**2).reshape(-1, 3), axis=0 )
        )
        depth_mean_stats.update(
            np.mean(dep),
            n=1
        )
        depth_square_stats.update(
            np.mean( dep**2 )
        )
    rgb_mean = rgb_mean_stats.avg
    depth_mean = depth_mean_stats.avg
    rgb_std = rgb_square_stats.avg - rgb_mean**2
    depth_std = depth_square_stats.avg - depth_mean**2

    return rgb_mean, rgb_std, depth_mean, depth_std

def collect_files(args):

    color_files = []
    depth_files = []

    # get the scene idx list
    scene_idx_list = read_numList_from_file(
        os.path.join(args.dataset_dir, "train.txt")
    )

    # parse the scene index and then the color image files
    for idx in scene_idx_list:
        color_folder = os.path.join(args.dataset_dir, str(idx), "color_images/")
        depth_folder = os.path.join(args.dataset_dir, str(idx), "depth_raw/")

        #if not os.path.exists(color_folder):
        #    continue

        num_cams = len(os.listdir(color_folder))
        for i in range(num_cams):
            color_f = os.path.join(color_folder, "color_image_{}.png".format(i))
            depth_f = os.path.join(depth_folder, "depth_raw_{}.npy".format(i))
            color_files.append(color_f)
            depth_files.append(depth_f)
    
    return color_files, depth_files


def main():
    args = get_args()

    # get files
    color_files, depth_files = collect_files(args)

    # compute stats
    rgb_means, rgb_stddev, depth_mean, depth_stddev = get_mean_std_files(color_files, depth_files)

    # print
    print("\n")
    print("The total number of training data: {}".format(len(color_files)))
    print("The rgb means are: {}".format(rgb_means))
    print("The rgb stddev are: {}".format(rgb_stddev))
    print("The depth mean is: {}".format(depth_mean))
    print("The depth stddev is: {}".format(depth_stddev))

if __name__=="__main__":
    main()
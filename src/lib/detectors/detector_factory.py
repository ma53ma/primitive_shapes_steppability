from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .multi_pose import MultiPoseDetector
from .grasp_pose import GraspPoseDetector

detector_factory = {
  'multi_pose': MultiPoseDetector, 
  "grasp_pose": GraspPoseDetector
}

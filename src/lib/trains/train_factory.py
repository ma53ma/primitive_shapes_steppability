from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .multi_pose import MultiPoseTrainer
from .grasp_pose import GraspPoseTrainer

train_factory = {
  'multi_pose': MultiPoseTrainer, 
  'grasp_pose': GraspPoseTrainer
}

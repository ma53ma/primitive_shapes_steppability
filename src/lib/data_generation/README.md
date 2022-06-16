# Primitive Shape Grasp Dataset Generation

Author: Yiye Chen

Build on top of Yunzhi's [primitiveShapes_data_generation](https://github.gatech.edu/ylin466/primitiveShapes_data_generation) repository.

Previous repository only generate the object shape segmentation mask, whereas this repository will generate the grasp annotations for the 3d grasp detection research.

More specifically, the generated data and labels will be (the one kept from the previous repository is marked as the PSCNN):

- Data: 
  - RGB-D image (PSCNN)
  - Raw depth map (npy) (PSCNN)
- Labels: 
  - Binary segmentation mask (PSCNN)
  - Segmentation labels (PSCNN)
  - 3d Grasp annotations
  - Camera Pose annotations
  - Need to determine the format for the pose

Some code borrowed from the [Acronym](https://github.com/NVlabs/acronym). 


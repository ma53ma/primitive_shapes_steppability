# 3D Grasp

Author: Yiye Chen

The repository for the 3D grasp detection based on the 2D/2.D input.



# Structure

```data_generation/```: The code for the primitive shape grasp dataset generation

```data```: The directory for storing the generated data



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
  
    

## Run the code

Configure the scene parameters in the ```configs/ps_grasp.yaml```, and run the file:

```bash
python main_data_generate.py 
```



For the single-object scene generation, a configuration file is already created. Run:

```bash
python main_data_generate.py --config_file lib/data_generation/ps_grasp_single.yaml
```





## Acknowledgement

Some code borrowed from the [CenterPose](https://github.com/NVlabs/CenterPose.git) and [Acronym](https://github.com/NVlabs/acronym). 



## TODO:

- [x] The grasp collision check during the data generation seems weird. Need to fix it immediately
- [ ] Some objects might be too close to each other. Good to write an option to set a minimum distance.
- [x] The object can not be too small, or the grasp keypoints would be really close to each other. Tune the object size during the data generation
  - [ ] Ring
  - [ ] Bowl
  - [ ] Stick
- [ ] For the planarPnP, the special case when the null space contains multiple dimensions has not been dealt with. Current just return None
- [ ] The scene 145 in the ps_grasp_single_500 is weird
- [ ] The orientation-classification-based methods
  - [x] The option and the task heads 
  - [ ] The decoder
  - [ ] The loss (trainer)
- [ ] The L2 loss for the open width prediction. Use L2 because there is tolerance on the open width. So we don't need all of the predictions to be exactly the same as the ground truth, but want all of them to be close. L2 "pays more attention" to the large deviates, hence use the L2 

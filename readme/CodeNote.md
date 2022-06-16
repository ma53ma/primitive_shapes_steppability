# Code Structure Notes



## Center Net

### Model - ```src/lib/models```

```src/lib/models``` defines the basic network models, including:

- Various backbones (hourglass, dla, resnet, etc) 
- Losses functions (look into that)
- Network output decoding functions (look into that!)
- Data parallel processing functions (useless for me)



### Detector -  ```src/lib/detectors``` 

```src/lib/detectors``` build on top of the ```models```. It defines the processing pipeline.

#### Model:

Stored as the ```self.model``` in the detector. It is an instance of the network classes. (e.g. ```DLASeg class```). All notes below are for the ```pose_dla_dcn``` code.

The network contains:

1. The feature extractor (e.g. ```DLASeg.ida_up```)
   - [ ] Check the structure of the DLASeg
     - First ```DLAup``` will get a pyramid of 4 feature maps, with the largest resolution ```1/4``` of the original size, and ```1/8```, ```1/16```, ```1/32```.
       - [ ] The structure of the DLAup
     - Then ```IDAup```, using the deformable convolution, get 3 feature maps, all of which has the ```1/4``` resolution.
       - [ ] The structure of the IDAup
     - Only the last one is taken as the final feature map.
2. The task heads (e.g. ```DLASeg.heads```)
   - The task heads are all 2-convolution-layer small network with the structure:
     - conv(feature dim -> middle_dim, kernel_size=3, padding=1, stride=1)
     - relu
     - conv(middle_dim -> class_num, kernel_size=1, padding=0, stride=1)
   - The task is created from the ```opt.head``` and the ```opt.head_conv```. 
     - The ```opt.head``` is a dictionary of ```task_name: class_num``` (e.g. ```hm_hp: 17```)
     - The ```opt.head_cov``` is the middle dim. If <0, then will skip that conv and relu.

#### Base Detector:

- ```Run``` function defines the pipeline to deal with one single image: ```multiscale (preprocess -> process -> postprocess) -> merge multiscale detection to get the final results -> show_results if set to the correct debug mode ```

- ```Preprocess``` defines the data preprocessing methods depends on the input parameters, including (in the order):
  - Scaling (Optional. For the multiscale testing) 
  
  - Padding
  
  - Normalization (divide by 255, zero mean, unit variance) - ***using dataset mean and variance during training, and what during testing?***
  
  - Flip. (Optional, for the *flipping test*) - **Flip along the width direction **
  
    **NOTE**: The preprocess here is only for the test data. It is less complicated than the data augmentation in the training, which is written in the datasets' ```__getitem__``` function.
  
- ```process```, ```postprocess```, ```merge_outputs (multiscale)```, ```show_results``` to be defined by child detectors.

- ```flip_test``` is to flip the image along the width direction (left-right) and to see whether can get consistent results. For each test image, itself and the flipped version of itself (Batch size is doubled) will be sent together into the model, and the result is somehow averaged to see whether consistent results can be obtained.

### Dataset & Dataloader

The dataset is split into two classes: The dataset and the the sampler. 

The dataset defines how to read the data and stores into a list. The sampler defines how to 

#### Data augmentation:

- The input resolution is resized to 512-by-512, hence the output encoding map is 128-by-128
- Augmentation strategy involves: random flipping, random scaling, cropping, and color jittering
  - Note that the input label should also be augmented
- *No augmentation is applied for the 3d estimation, as the 3d measurements will be changed by the scaling or cropping* 



### Trainer

The functionality is to defining the training pipeline. The Base trainer takes the model, the optimizer, and the dataloader and defines the process of training/evaluating the model for one epoch. 

Each child trainer for different tasks need to define:

- Losses in the ```self._get_losses() ``` function. This function needs to return:
  - loss_stat (List(str)): The names of the losses
  - loss (exec): The executable that takes in network outputs and the dataset batch, and returns a dictionary of losses indexed by the *loss_stat* above,

**NOTE**:

- The trainer takes the model, not the detector. Hence the output label from the training set needs to be for the model output.
- The detector will take the output, and postprocess to recover the result in the original image space.

#### Losses

A break down of losses in the ``multi_pose`` case, (i.e. the ```MultiPoseLoss``` class in the multi_pose trainer):

1. Focal loss for the heatmaps. Takes in a 



### Training

- **Main.py**: The shared training main file:

- For training the DLA network on the multi_pose task, run the following command instead of the script in the experiment folder (because they use 8 GPUs):

  ```bash
  python main.py multi_pose --exp_id dla_3x --dataset coco_hp --batch_size 8 --master_batch 9 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0 --num_workers 4 --num_epochs 320 --lr_step 270,300
  
  ```



### Testing

Run the ```test.py```. For the human_pose:

```bash
python test.py multi_pose --exp_id dla --dataset coco_hp --keep_res --load_model ../models/multi_pose_dla_3x.pth --flip_test
```





## PSGrasp 

### Summary

If want to create new the model for a new task:

- [x] Dataset loader
  - [x] New dataset
  - [x] New sampler
  - [x] Define the evaluation metric
- [ ] Create a new model with the new task heads

- [ ] Create a new detector that decode the output of the new model
- [ ] A new trainer

### Dataset

If want to set a canonical dataset (for both training or testing), use the option: ```--open_width_canonical VALUE```

If want to set a minimum open width (which will only set the width below to that width, for both training and testing), use the option: ```--min_open_width VALUE```



### Training

Run:

```bash
python main.py grasp_pose --exp_id grasp_train_single_1k --dataset ps_grasp --batch_size 12 --lr 1.25e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0 --num_workers 4 --num_epochs 400 --lr_step 350,370 --pnp_type cvIPPE --kpt_type box --open_width_canonical 0.1 --no_nms --correct_rl --ori_clf_mode --ori_num 9 --no_kpts_refine
```



If needs debug can add the following option:

```--bebug 4```: Will save the GT results and the detection results to a folder

```--inspect_aug ```: Will display the results in the data augmentation process

Hence the script:

```bash
python main.py grasp_pose --exp_id grasp_train_unitTest --dataset ps_grasp --batch_size 1 --lr 1.25e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0 --num_workers 4 --num_epochs 320 --lr_step 270,300  --pnp_type cvIPPE --kpt_type box --open_width_canonical 0.1 --no_nms --debug 4 --inspect_aug --correct_rl --ori_clf_mode --ori_num 9
```

To add the vanishing kpt loss, add the option:

```--vpt_loss_center_weight 1e-6 --vpt_loss_kpts_weight 1e-6``` 



Training unit test:

Remove the randomness in the data augmentation

```bash
python main.py grasp_pose --exp_id grasp_train_unitTest --dataset ps_grasp --batch_size 1 --lr 1.25e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0 --num_workers 4 --num_epochs 320 --lr_step 270,300  --pnp_type cvIPPE --kpt_type box --open_width_canonical 0.1 --no_nms --unitTest --debug 4 --flip 0 --no_color_aug --fix_crop --correct_rl --ori_clf_mode --ori_num 9 
```



#### Training Unit test

Due to the dense nature of the keypoint heatmaps, the NMS has to be removed!





### Testing

Note: The following options:

 - Removes the flip test (```--flip_test```)
 - Not using the prefetch_test (```--not_prefetch_test```)

Run:

```bash
python test.py grasp_pose --exp_id grasp_test_10k --dataset ps_grasp --keep_res --load_model ../exp/grasp_pose/grasp_train_single_10k/model_best.pth --not_prefetch_test --trainval --kpt_type box --pnp_type cvIPPE --center_thresh 0.5 --vis_thresh 0.5 --kpts_hm_thresh 0.1 --open_width_canonical 0.1 --no_nms --rot_sample_num 30 --trl_sample_num 10 --ori_clf_mode --ori_num 9
```

Options:

- ```--debug 5```. Save out the results 
- ```--test_num_per_shape 10```. Only test a certain number of samples per shape.



If want to use the GT kpt projection:

```bash
python test.py grasp_pose --exp_id grasp_test_gt --dataset ps_grasp --keep_res --not_prefetch_test --test_oracle_kpts --trainval --kpt_type box --pnp_type cvIPPE
```



Test with visualization on a subset of data:

```bash
python test.py grasp_pose --exp_id grasp_test_debug --dataset ps_grasp --keep_res --load_model ../exp/grasp_pose/grasp_train_single_1k_RL_oriClf9/model_last.pth --not_prefetch_test --trainval --kpt_type box --pnp_type cvIPPE --center_thresh 0.5 --vis_thresh 0.5 --kpts_hm_thresh 0.1 --open_width_canonical 0.1 --no_nms --debug 5 --vis_results --test_num_per_shape 20 --ori_clf_mode --ori_num
```





### Demo

The multi_pose demo:

```bash
python demo/demo.py multi_pose --demo /home/cyy/Research/Grasp/3d_grasp/images/17790319373_bd19b24cfc_k.jpg --load_model /home/cyy/Research/Grasp/3d_grasp/models/multi_pose_dla_3x.pth --debug 2
```



The grasp demo:

```bash
python demo/demo.py grasp_pose --dataset ps_grasp --keep_res --correct_rl --ori_clf_mode --ori_num 9  --demo /local/dropbox_cyy/Dropbox\ \(GaTech\)/GraspExpTRO/SingleObj/Banana/30 --load_model /home/cyy/Research/Grasp/3d_grasp/exp/grasp_pose/grasp_train_single_1k_RL_oriClf9/model_last.pth --no_nms --kpt_type box --pnp_type cvIPPE --center_thresh 0.2 --vis_thresh 0.2 --kpts_hm_thresh 0.1 --debug 2
```





# Baseline - GPG

Test:

```bash
python test_baselines.py grasp_baseline_GPG --exp_id grasp_test_GPG --dataset ps_grasp --trainval --not_prefetch_test --rot_sample_num 30 --trl_sample_num 10 --using_mp
```

Additional options:

```--load_exist_baseResults```: Use the pre-saved GPG detection results.

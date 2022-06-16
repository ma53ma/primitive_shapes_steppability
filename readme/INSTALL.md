# Installation

Slighty modified from the CenterNet installation guide. Install the [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4) externally instead of within the repository. Tested on Ubuntu 16.04 with Python3.7 and [PyTorch]((http://pytorch.org/)) v1.4.0. NVIDIA GPUs are needed for both training and testing.


After install Anaconda:

0. [Optional but recommended] create a new environment. 

    ~~~
    python3.7 -m venv 3d_grasp
    ~~~
    And activate the environment.
    
    ~~~
    source 3d_grasp/bin/activate
    ~~~

1. Install pytorch1.4.0:

    ~~~
    pip install torch==1.4.0 torchvision==0.5.0
    ~~~
    
    Manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. We observed slight worse training results without doing so.
    
    For the pytorch v0.4.0 and v0.4.1, disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).
    
     ~~~
    # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
    # for pytorch v0.4.0
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    # for pytorch v0.4.1
    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
     ~~~
    
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo:

    ~~~
    CenterNet_ROOT=/path/to/clone/CenterNet
    git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
5. Install and Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)). **Choose the tag according to the pytorch version. (for this repository we use the pytorch_1.0**

    ~~~
    # DCNv2=/path/to/clone/DCNv2
    git clone https://github.com/CharlesShang/DCNv2.git $DCNv2
    cd $DCNv2
    git checkout tags/pytorch_1.0
    ./make.sh
    python setup.py install --user
    ~~~
    
6. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

    ~~~
    cd $CenterNet_ROOT/src/lib/external
    make
    ~~~




# Install for the GPG

Modified and simplified from the [PointNetGPD](https://github.com/lianghongzhuo/PointNetGPD#install-all-the-requirements-using-a-virtual-environment-is-recommended). The original installation method of the ```python-pcl``` can not work, so here the package is installed system-wise, and a separate virtual environment is created to inherit the package (step 1). On the Ubuntu 20.04, the default python version is 3.8.

1. Install `pcl-tools` 

   ```bash
   sudo apt install libpcl-dev
   sudo apt install python3-pcl
   python3 -m venv pnGPD --system-site-packages
   ```

1. Create a virtual environment and inherit the system site packages:

   ```bash
   python3 -m venv pnGPD --system-site-packages
   source pnGPG/bin/activate
   ```

1. `pip install mayavi`

1. Install PyTorch: https://pytorch.org/get-started/locally/

   ```bash
   pip install torch==1.4.0 torchvision==0.5.0
   ```

1. Install the PointNetGPD requirements in `requirements_pnGPD.txt`

   ```bash
   cd $PointNetGPD_FOLDER
   pip install -r requirements_pnGPD.txt
   ```

1. Install the PointNetGPD modified meshpy (Modify from [Berkeley Automation Lab: meshpy](https://github.com/BerkeleyAutomation/meshpy))

   ```bash
   cd $PointNetGPD_FOLDER/meshpy
   python setup.py develop
   ```

1. Install the PointNetGPD modified dex-net (Modify from [Berkeley Automation Lab: dex-net](https://github.com/BerkeleyAutomation/dex-net))

   ```bash
   cd $PointNetGPD_FOLDER/dex-net
   python setup.py develop
   ```


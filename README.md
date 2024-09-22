# YOLO.V5-on-Jetson-Nano and CSI-CAMERA

# What is YOLO? 
YOLO (You Only Look Once) is a real-time object detection algorithm developed by Joseph Redmon and Ali Farhadi in 2015. It is a single-stage object detector that uses a convolutional neural network (CNN) to predict the bounding boxes and class probabilities of objects in input images.

# What is Jetson-nano?
The Jetson Nano module is a small AI computer that gives you the performance and power efficiency to take on modern AI workloads, run multiple neural networks in parallel, and process data from several high-resolution sensors simultaneously.

# How to install Yolo on Jetson Nano?

Before I start to expalin how to run Yolo on jetson, I have to mention that the last version of Jetpack is 4.6.1 which consists of Python 3.6.9. However, Yolo family(Ultrlytics) works with python>=3.8. Furthermore, some of Yolo requirments package are not developed for arm64. As a result, in this tutorial, some part of our work fucos on installing them.

1) By default Jetpack comes with cuda 10.2. However, nvcc command are not recognised. To address this issue you have to add cuda to ubunto Environment path by bellow commands:
   
   1-1) sudo apt-get install nano
   
   1-2) nano ~/.bashrc
   
   1-3) at the end of file add bellow command:
   
   1-4) export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   
   1-5) ~/.bashrc
   
   1-6) if everything work well you have to see cuda version by running <<nvcc --version>> on the terminal.
   
3) Install torch and torch vision on Jetson Nano
   
   2-1) Go to https://forums.developer.nvidia.com/t/pytorch-for-jetson
   
   2-2) you can see all versions of pytorch on the jetpack 4, copy the name of torch version
   
   2-3) wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O << name of torch version>>
   
   2-3-1) for excample: wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O  torch-1.10.0-cp36-cp36m-linux_aarch64.whl
   
   2-4) sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
   
   2-5) pip3 install 'Cython<3'
   
   2-6) pip3 install <<name of torch version>> for example: pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
   
5) Install torchvision compatible with Torch:
   
   3-1) to install torchvision you have to fid the compatible version of torchvision with your torch, to address this issue, on the webpage all version of torch and its torchvision are mentioned.
   
   3-2) sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
   
   3-3) git clone --branch <version of torchvision> https://github.com/pytorch/vision torchvision
   
   3-3-1) for example: git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
   
   3-4) cd torchvision
   
   3-5) export BUILD_VERSION=0.x.0 where 0.x.0 is the torchvision version( for example 0.11.1)
   
   3-6) python3 setup.py install --user
   
   3-7) cd ../
   
   3-8) pip install 'pillow<7'
   
7) Verification Torch and TorchVision

   7-1) python3
   
     import torch
     print(torch.__version__)
     print('CUDA available: ' + str(torch.cuda.is_available()))
     print('cuDNN version: ' + str(torch.backends.cudnn.version()))
     a = torch.cuda.FloatTensor(2).zero_()
     print('Tensor a = ' + str(a))
     b = torch.randn(2).cuda()
     print('Tensor b = ' + str(b))
     c = a + b
     print('Tensor c = ' + str(c))

5) after installing torch and torchvision we are going to install Yolo and its dependencies.
   
  5-1) clone our repository
  
  5-2) on this repository we only used yolov5n if you want to work with other versions you have to download the weights from
  
  5-2-1) https://github.com/ultralytics/yolov5/releases/tag/v6.2
  
  5-2-2) download the desired verion on yolo under the asset part

6) after cloning the repository you can see Yolo requirements on requirments.txt. However, some of its dependencies does not install with pip
   
  6-1) pip3 install tqdm
  
  6-2) pip3 install psutil
  
  6-3) pip3 install thop
  
  6-4) sudo apt install python-seaborn

7) Run python3 version1.py to run Yolo on CSI Camera

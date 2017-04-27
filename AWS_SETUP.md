# AWS Setup

m4.large for download
p2.xlarge for training
300GB SSD Disk

http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/
http://virtualenvwrapper.readthedocs.io/en/latest/install.html

sudo apt upgrade
sudo apt upgrade

sudo apt install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran libblas-dev liblapack-dev libatlas-base-dev python-dev linux-headers-generic linux-image-extra-virtual unzip swig wget pkg-config zip g++ zlib1g-dev libcurl3-dev python3-dev python3-pip

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

pip2 install virtualenv
pip2 install --upgrade pip
sudo pip2 install virtualenvwrapper

in ~/.bashrc
export WORKON_HOME=/home/ubuntu/virtualenvs
source /usr/local/bin/virtualenvwrapper.sh

source ~/.bashrc

mkvirtualenv kaggle -p `which python3`

wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
rm cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install -y cuda


pip install tensorflow-gpu
pip install keras
pip install pandas
pip install matplotlib
pip install seaborn
pip install sklearn
pip install scikit-image
pip install opencv-contrib-python
pip install shapely
pip install h5py
pip install ipython
pip install jupyter



sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
rm cudnn-8.0-linux-x64-v5.1.tgz 
rm -rf cuda/


ubuntu@ip-172-31-1-174:~$ nvidia-smi
Thu Apr 27 06:14:56 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.39                 Driver Version: 375.39                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 0000:00:1E.0     Off |                    0 |
| N/A   51C    P0    59W / 149W |      0MiB / 11439MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

in ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64


mkdir libs
cd libs
git clone https://github.com/tensorflow/tensorflow.git
python tensorflow/tensorflow/examples/tutorials/mnist/mnist_deep.py

Checks:
nvidia-smi
ipython: import keras, import cv2


# RPSnet Implementation

Implementation of a saliency map-based interpretability algorithm for use with Random Path Selection for Incremental Learning (RPSnet).

[//]: # (Short Description of Algorithm)

This code provides an implemenation of a novel continual learning interpretability algorithm which utilizes 
saliency maps. This folder contains the implementation for iTAML; the algorithm was implemented using PyTorch.

## Installation & Requirements
This code was has been tested using `Python 3.6.8` on a CentOS operating system. 

### Dependencies
The following dependencies are required to run this code:
```
numpy==1.19.5
matplotlib==3.3.4
pandas==1.1.5
Pillow==8.4.0
scipy==1.4.1
torch==1.10.1
torchvision==0.11.2
tensorflow==2.2.0
keras==2.3.1
captum==0.7.0
```
These dependencies can be installed by running the following command:
```
pip install -r requirements.txt
```

## Usage
To run the code the following command can be used:
```
./DATASET.sh
```
Here `DATASET` should be replaced with the desired dataset for testing. For example, in order to run the MNIST dataset the following command should be used:
```
./MNIST.sh
```

## Credit
We credit https://github.com/brjathu/RPSnet for the original RPSnet algorithm.

[//]: # (### Acknowledgements)
# Gaze-Segment-Anything-YCB-Video
This project uses gaze information from VR-Headset to segment YCB objects from images. 

## Installation
The code requires `python>=3.8`, `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.


First create a conda environment:
```
conda create --name sam-ycb
conda activate sam-ycb
```

Install SegmentAnything and its dependencies (for mask post-processing):
```
pip install git+https://github.com/facebookresearch/segment-anything.git

pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Install Yolov8:
```
pip install ultralytics
```
## Experiments

### Train Yolo
For this project, we will first train Yolo to detect YCB objects on the images. We will then train it on the YCB dataset with the `train_yolo.py` script.
First of all, download the YOLOv8n model that we will fine-tune on YCB-video dataset : [download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt).

To launch the script you can use the following command line : 
```
```

### Launch Segmentation 
Now that Yolo is trained, we will combine its resulting bounding box to extract the associated segmentation mask with the Segment Anything Model. To do so, launch the `ycb_segment.py` script by using the following command line :

```
```

### Complete with gaze information
The final step of the project is to further combine the information from the bounding box AND a point gaze information to lock an object in a scene and return its segmentation mask.

This mask will then be used as input to the 6D pose estimation model DenseFusion to extract the pose of the detected object. 

## Future work and ideas
It would be interesting to further combine SAM with depth as it was done in [SegmentAnyRGBD](https://github.com/Jun-CEN/SegmentAnyRGBD).

## References
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
[Yolov8](https://github.com/ultralytics/ultralytics)
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
## Purpose of the project
Segment Anything has demonstrated incredible zero-shot generalization capabilities to predict segmentation of objects on images given prompts. YOLOv8 model is a very powerful model for object detection that can also be used for object segmentation. However, while its training for object-detection is really easy to setup, its segmentation training is cumbersome because it needs a specific format not directly applicable to other downstream tasks. 

In this project, we want to combine the capacity of YOLOv8 to detect objects with the incredible capacities of SAM to segment objects given a prompt to detect and segment objects from a given dataset (here the YCB-video dataset).

This base project would then be used in combination with VR-Headset gaze in order to extract the objects of interest that a user is looking at.

## Experiments

### Pre-process the dataset
We first need to pre-process the YCB-Video dataset in order to make it possible to train YOLO on it. We follow the formatting given [here](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#train-on-custom-data) to format the dataset. To launch the pre-processing, launch the following script :
```
python3 process_ycb.py --dataset_path <PATH_TO YCB_Video_Dataset FOLDER> --data_config <PATH_TO_CONFIG_YAML_FILE>
```

When launched this script takes a long time (~50 minutes), so go grab a cup of coffee while it is running. 

### Train Yolo
Once the data are ready, we can begin to train YOLO to detect YCB objects on the images. We will then train it on the YCB dataset with the `train_yolo.py` script.
First of all, download the YOLOv8n model that we will fine-tune on YCB-video dataset : [download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt).

To launch the script you can use the following command line : 
```
python3 train_yolo.py --model_path <PATH_TO_THE_MODEL> --data_config <PATH_TO_CONFIG_YAML_FILE>
```

### Launch Segmentation 
Now that Yolo is trained, we will combine its resulting bounding box to extract the associated segmentation mask with the Segment Anything Model. To do so, launch the `ycb_segment.py` script by using the following command line :

```
python3 ycb_segment.py --image_folder <PATH_TO_FOLDER_OF_IMAGES> --yolo_path <PATH_TO_YOLO_MODEL>
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
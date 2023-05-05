import argparse
from ultralytics import YOLO
import torch


"""
This script permits to launch the training of the pre-trained YOLO model on the YCB-Video dataset. 
"""
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Extract the segmentation of objects on YCB dataset with Yolo bounding box'
    )
    parser.add_argument('--model_path', type=str, help='Path to the pretrained YOLO model')
    parser.add_argument('--data_config', type=str, help='.yaml config file used to train the model')
    
    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model_path) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model.to(device)

    # Use the model
    model.train(data=args.data_config, batch=32, epochs=15)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
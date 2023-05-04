import argparse
from ultralytics import YOLO


"""
This script permits to launch the training of the pre-trained YOLO model on the YCB-Video dataset. 
"""
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Extract the segmentation of objects on YCB dataset with Yolo bounding box'
    )
    parser.add_argument('--model_path', type=int, help='Path to the pretrained YOLO model')
    parser.add_argument('--data_config', type=int, help='.yaml config file used to train the model')
    
    args = parser.parse_args()


    # Load a model
    model = YOLO(args.model_path)  # load 

    # Use the model
    model.train(data=args.data_config, epochs=40)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
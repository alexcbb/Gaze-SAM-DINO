from PIL import Image 
import os, sys, argparse
from os import listdir
from os.path import isfile, join
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2
import numpy as np

def bbox_unnormalize(bbox, img_width, img_height):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    w = w * img_width
    h = h * img_height
    x = x * img_width
    y = y * img_height
    return x, y, w, h

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Extract the segmentation of objects on YCB images with Yolo bounding box'
    )
    parser.add_argument('--image_folder', type=str, help='Path to a folder containing images to test')
    parser.add_argument('--yolo_path', type=str, help='Path to the trained YOLO model')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Load SAM
    #sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth") # To use this model, you need to download it here : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    #sam = sam.to(device)
    #predictor = SamPredictor(sam)

    # Load YOLO
    yolo = YOLO(args.yolo_path)
    yolo.to(device)

    images = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]

    for image_path in images:
        start_time = datetime.now()

        # Open image
        image_path = args.image_folder + "/" + image_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict bounding boxes
        results = yolo.predict(image)
        results = results[0]
        #boxes = torch.tensor(res[0].boxes)#, device=predictor.device)

        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image)

        # Create a Rectangle patch
        for obj_ in res[0].boxes:
            obj = obj_.xywh[0].cpu()
            color = np.random.rand(3) # generate a random color
            rect = plt.Rectangle((obj[0], obj[1]), obj[2], obj[3], linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        """ for box in res[0].boxes[0]:
            box_new = box.xywh[0].cpu()
            w, h = box_new[2], box_new[3]
            x, y = box_new[0]- w /2, box_new[1] - h/2
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)"""

        plt.show()

        # Extract SAM predictions with bounding boxes
        """predictor.set_image(image)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        end_time = datetime.now()
        time_difference = (end_time - start_time).total_seconds() * 10**3
        print("Inference time: ", time_difference, "ms")

        print(masks)
        print(masks.shape)"""





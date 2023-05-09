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


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

# Function obtained from : https://inside-machinelearning.com/en/bounding-boxes-python-function/
def plot_bboxes(image, boxes, score=True, conf=None):
    labels = {0: u'002_master_chef_can', 1: u'003_cracker_box', 2: u'004_sugar_box',3: u'005_tomato_soup_can', 4: u'006_mustard_bottle', 5: u'007_tuna_fish_can', 6: u'008_pudding_box', 7: u'009_gelatin_box', 8: u'010_potted_meat_can', 9: u'011_banana', 10: u'019_pitcher_base', 11: u'021_bleach_cleanser', 12: u'024_bowl', 13: u'025_mug', 14: u'035_power_drill', 15: u'036_wood_block', 16: u'037_scissors', 17: u'040_large_marker', 18: u'051_large_clamp', 19: u'052_extra_large_clamp', 20: u'061_foam_brick'}
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57)]
    
    #plot each boxes
    for box in boxes:
        #add score in label if score=True
        if score:
            label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else:
            label = labels[int(box[-1])]

        #filter every box under conf threshold if conf threshold setted
        if conf :
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)
    return image

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Extract the segmentation of objects on YCB images with Yolo bounding box'
    )
    parser.add_argument('--image_folder', type=str, help='Path to a folder containing images to test')
    parser.add_argument('--yolo_path', type=str, default="best.pt", help='Path to the trained YOLO model')
    parser.add_argument('--sam_path', type=str, default="sam_vit_h_4b8939.pth", help='Path to the SAM model')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Load SAM
    """sam = sam_model_registry["default"](checkpoint=args.sam_path) # To use this model, you need to download it here : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    sam = sam.to(device) 
    predictor = SamPredictor(sam)"""

    # Load YOLO
    yolo = YOLO(args.yolo_path)
    yolo.to(device)

    # Extract all images name in the image folder
    images = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]

    for image_name in images:
        start_time = datetime.now()

        # Open image
        image_path = args.image_folder + "/" + image_name
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()

        # Predict bounding boxes
        results = yolo.predict(image)

        # Set the image as input to SAM
        #predictor.set_image(image)
        #results = results[0]
        #boxes = torch.tensor(res[0].boxes)#, device=predictor.device)

        # Create a Rectangle patch
        """for res in results:
            for box in res.boxes:
                box_pose = box.xywh[0].cpu()
                color = np.random.rand(3) # generate a random color
                rect = plt.Rectangle((box_pose[0], box_pose[1]), box_pose[2], box_pose[3], linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)"""
        for res in results:
            image = plot_bboxes(image, res.boxes.boxes)
            print(image.shape)
            plt.imshow(image)
            plt.show()

        """# Extract SAM predictions with bounding boxes
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




